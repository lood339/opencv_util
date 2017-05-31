//
//  cvx_calib2d.cpp
//  CalibMeMatching
//
//  Created by jimmy on 2017-05-23.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "cvx_calib2d.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#include "lsd_line_segment.h"
#include "cvx_imgproc.hpp"
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "vnl_algo_homography.h"
#include <iostream>
#include <opencv2/plot.hpp>
#include "eigen_matlab_writer.h"

namespace cvx {
    // from opencv
    /*
     p: confidence
     ep: current inlier percentage
     modelPoints: minimum number of points required for the solution
     maxIters: current set of maximum iteration number
     */
    
    using cv::Mat;
    using cv::Point2f;
    using cv::Point2d;
    using cv::plot::Plot2d;
    using std::cout;
    using std::endl;
    
    
    int RANSACUpdateNumIters( double p, double ep, int modelPoints, int maxIters )
    {
        if( modelPoints <= 0 )
            CV_Error( cv::Error::StsOutOfRange, "the number of model points should be positive" );
        
        p = MAX(p, 0.);
        p = MIN(p, 1.);
        ep = MAX(ep, 0.);
        ep = MIN(ep, 1.);
        
        // avoid inf's & nan's
        double num = MAX(1. - p, DBL_MIN);
        double denom = 1. - std::pow(1. - ep, modelPoints);
        if( denom < DBL_MIN )
            return 0;
        
        num = std::log(num);
        denom = std::log(denom);
        
        return denom >= 0 || -num >= maxIters*(-denom) ? maxIters : cvRound(num/denom);
    }
    
    bool isHomographyInlier(const cv::Point2f& src_p, const cv::Point2f& dst_p,
                            const cv::Mat& H, double reproj_threshold)
    {
        assert(H.type() == CV_64FC1);
        
        cv::Mat p = cv::Mat::zeros(3, 1, CV_64FC1);
        p.at<double>(0, 0) = src_p.x;
        p.at<double>(1, 0) = src_p.y;
        p.at<double>(2, 0) = 1.0;
        
        cv::Mat proj_p = H * p;
        double z = proj_p.at<double>(2, 0);
        if ( fabs(z) <= 0.0000001) {
            return false;
        }
        
        double x = proj_p.at<double>(0, 0)/z;
        double y = proj_p.at<double>(1, 0)/z;
        
        cv::Point2f dif = dst_p - cv::Point2f(x, y);
        double dist = cv::norm(dif);
        if (dist < reproj_threshold) {
            return true;
        }
        else {
            return false;
        }
    }

    vector<Mat> findHomography(const vector<vector<cv::Point2f> >& src_point_groups,
                               const vector<vector<cv::Point2f> >& dst_point_groups,
                               vector<unsigned char>& mask,
                               int method,
                               double ransac_reproj_threshold,
                               const int max_iters,
                               const double confidence)
    {
        assert(src_point_groups.size() == dst_point_groups.size());
        assert(src_point_groups.size() == 2);
        assert(src_point_groups[0].size() > 4);
        
        const int N = (int)src_point_groups[0].size();  // sample number
        const int M = (int)src_point_groups.size();     // model number
        int best_inlier_num = 0;
        vector<Mat> best_result;
        int niter = max_iters;
        mask.resize(N, 0);
        for (int iter = 0; iter < niter; iter++) {
            // step 1. randomly select four groups
            int init_index[4] = {0, 0, 0, 0};
            int max_random_iter = 100;
            while ((init_index[0] == init_index[1] || init_index[0] == init_index[2] || init_index[0] == init_index[3] ||
                    init_index[1] == init_index[2] || init_index[1] == init_index[3] || init_index[2] == init_index[3])
                    && max_random_iter--) {
                for (int i = 0; i<4; i++) {
                    init_index[i] = rand()%N;
                }
            }
            
            vector<vector<cv::Point2f> > rnd_src_points(M);
            vector<vector<cv::Point2f> > rnd_dst_points(M);
            for (int i = 0; i<M; i++) {
                for (int j = 0; j<4; j++) {
                    int index = init_index[j];
                    rnd_src_points[i].push_back(src_point_groups[i][index]);
                    rnd_dst_points[i].push_back(dst_point_groups[i][index]);
                }
            }
            
            // step 2. estimate init model
            vector<Mat> cur_models;
            for (int i = 0; i < M; i++) {
                Mat h = cv::findHomography(rnd_src_points[i], rnd_dst_points[i]);
                if (!h.empty()) {
                    cur_models.push_back(h);
                }
            }
            if (cur_models.size() < rnd_src_points.size()) {
                printf("warning: group init failed, valid homography number: %lu, should be %d.\n", cur_models.size(), M);
                continue;
            }
            
            assert(cur_models.size() == M);
            
            int inlier_num = 0;
            vector<int> inlier_indices;
            // step 3. update model
            for (int i = 0; i<N; i++) {
                bool is_inlier = true;
                for (int m = 0; m<cur_models.size(); m++) {
                    cv::Point2f p = src_point_groups[m][i];
                    cv::Point2f q = dst_point_groups[m][i];
                    if (!isHomographyInlier(p, q, cur_models[m], ransac_reproj_threshold)) {
                        is_inlier = false;
                        break;
                    }
                }
                if (is_inlier) {
                    inlier_num++;
                    inlier_indices.push_back(i);
                }
            }
            
            if (inlier_num > best_inlier_num) {
                best_inlier_num = inlier_num;
                
                vector<vector<cv::Point2f> > src_inliers(M);
                vector<vector<cv::Point2f> > dst_inliers(M);
                for (int m = 0; m<M; m++) {
                    for (int i = 0; i<inlier_indices.size(); i++) {
                        int index = inlier_indices[i];
                        src_inliers[m].push_back(src_point_groups[m][index]);
                        dst_inliers[m].push_back(dst_point_groups[m][index]);
                    }// i
                }// m
                
                cur_models.clear();
                for (int i = 0; i<M; i++) {
                    Mat h = cv::findHomography(src_point_groups[i], dst_point_groups[i]);
                    if (!h.empty()) {
                        cur_models.push_back(h);
                    }
                }
                if (cur_models.size() == M) {
                    best_result = cur_models;
                    
                    // update mask
                    mask.resize(N, 0);
                    for (int i = 0; i<inlier_indices.size(); i++) {
                        int index = inlier_indices[i];
                        mask[index] = 255;
                    }
                  
                    // step 4. update iteration number
                    double ep = 1.0 - 1.0 * inlier_indices.size()/N;  // current inlier percenrage
                    niter = RANSACUpdateNumIters(confidence, ep, 4, niter);
                    printf("number of iteration updated to %d\n", niter);
                }
            }
        }
        return best_result;
    }
    
    static double transformedPointLineDistance(const Eigen::Matrix3d& h, // transfromation matrix
                                               double x1, double y1,  // point before transformation
                                               double x2, double y2,  // line end point after transformation
                                               double x3, double y3)
    {
        Eigen::Vector3d p(x1, y1, 1.0);
        p = h * p;
        if (p.z() == 0) {   // specialy case
            cout<<"Warning: z in homogenerious coordinate is zero."<<endl;
            return INT_MAX;
        }
        p /= p.z();
        
        Eigen::Vector2d q1(x2, y2);
        Eigen::Vector2d q2(x3, y3);
        Eigen::ParametrizedLine<double, 2> line = Eigen::ParametrizedLine<double, 2>::Through(q1, q2);
        double dist = line.distance(Eigen::Vector2d(p.x(), p.y()));
        return dist;
    }
    
    
    bool findHomography(const cv::Mat& src_image,
                        const cv::Mat& dst_image,
                        const int method,
                        cv::Mat & output_warp,
                        double& warp_quality,
                        double search_length, int block_size)
    {
        assert(src_image.type() == CV_8UC1 && dst_image.type() == CV_8UC1);
        assert(src_image.size == dst_image.size);
        assert(method == 0 || method == 1);
        
        const int rows = src_image.rows;
        const int cols = src_image.cols;
        
        // step 1: tracking points
        cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,20,0.03);
        cv::Size subPixWinSize(10,10), winSize(31,31);
        
        const int max_count = 2000;
        const int min_point_distance = 0.02 * cols;    // space distance of sampled point features
        const double reproj_threshold = 2.0;
        const double edge_to_line_reproj_threshold = 1.5;
        vector<cv::Point2f> point_groups[2];
        cv::goodFeaturesToTrack(src_image, point_groups[0], max_count, 0.0001, min_point_distance, Mat(), 3, true, 0.04);
        cornerSubPix(src_image, point_groups[0], subPixWinSize, cv::Size(-1,-1), termcrit);
        
        vector<uchar> forward_status;
        vector<float> forward_err;
        cv::calcOpticalFlowPyrLK(src_image, dst_image, point_groups[0], point_groups[1],
                                 forward_status, forward_err, winSize,
                                 3, termcrit, 0, 0.001);
        
        // correctly tracked points
        vector<Point2f> tracked_point_groups[2];
        for (int i = 0; i<forward_status.size(); i++) {
            if (forward_status[i] != 0) {
                tracked_point_groups[0].push_back(point_groups[0][i]);
                tracked_point_groups[1].push_back(point_groups[1][i]);
            }
        }
        assert(tracked_point_groups[0].size() == tracked_point_groups[1].size());
        if (tracked_point_groups[0].size() < 10) {  // magic number 4 * 2 + 2
            return false;
        }
        
        // step 3: estimate homography using RANSAC
        vector<uchar> mask;
        Mat h = findHomography(tracked_point_groups[0], tracked_point_groups[1], CV_RANSAC, reproj_threshold, mask);
        if (h.empty()) {
            return false;
        }
        assert(h.type() == CV_64FC1);
        
        // using both points and lines
        // step 3. edge tracking
        Mat src_temp, dst_temp;
        src_image.convertTo(src_temp, CV_64FC1);
        if (!src_temp.isContinuous()) {
            src_temp = src_temp.clone();
        }
        dst_image.convertTo(dst_temp, CV_64FC1);
        if (!dst_temp.isContinuous()) {
            dst_temp = dst_temp.clone();
        }
        
        // step 3: detect lines and divide into short line segment
        std::vector<LSD::LSDLineSegment2D> lines1;
        std::vector<LSD::LSDLineSegment2D> lines2;
        LSD::detectLines((double *) src_temp.data, cols, rows, lines1);
        LSD::detectLines((double *) dst_temp.data, cols, rows, lines2);
        LSD::shortenLineSegments(lines1, 25);    // divide long lines to short lines
        LSD::shortenLineSegments(lines2, 25);
        
        vector<cv::Vec4f> src_lines(lines1.size());
        vector<cv::Vec4f> dst_lines(lines2.size());
        vector<cv::Vec2f> dst_centers;
        cv::Size sz(cols, rows);
        for (int i = 0; i<lines1.size(); i++) {
            src_lines[i] = cv::Vec4f(lines1[i].x1, lines1[i].y1, lines1[i].x2, lines1[i].y2);
        }
        for (int i = 0; i<lines2.size(); i++) {
            dst_lines[i] = cv::Vec4f(lines2[i].x1, lines2[i].y1, lines2[i].x2, lines2[i].y2);
        }
        
        cvx::trackLineSegmentCenter(src_lines, dst_lines, dst_centers, sz,
                                    cv::noArray(), cv::noArray(),
                                    search_length, block_size);        
        
        // step 4. homography using both points and lines
        assert(src_lines.size() == dst_centers.size());
        Eigen::Matrix3d inv_h;
        Eigen::Matrix3d h_eigen;
        cv2eigen(h.inv(), inv_h);
        cv2eigen(h, h_eigen);
        
        vector<Eigen::ParametrizedLine<double, 2> > inlier_src_lines;
        vector<Eigen::Vector2d> inlier_dst_line_pts;
        for (int i = 0; i<dst_centers.size(); i++) {
            Eigen::Vector3d p(dst_centers[i][0], dst_centers[i][1], 1.0);
            p = inv_h * p;
            p /= p.z();
            
            Eigen::Vector2d q1(src_lines[i][0], src_lines[i][1]);
            Eigen::Vector2d q2(src_lines[i][2], src_lines[i][3]);
            Eigen::ParametrizedLine<double, 2> line = Eigen::ParametrizedLine<double, 2>::Through(q1, q2);
            double dist = line.distance(Eigen::Vector2d(p.x(), p.y()));
            // inlier
            if (dist < edge_to_line_reproj_threshold) {
                inlier_src_lines.push_back(line);
                inlier_dst_line_pts.push_back(Eigen::Vector2d(dst_centers[i][0], dst_centers[i][1]));
            }
        }
        //cout<<"inlier edge number "<<inlier_src_lines.size()<<" . Percentage "<<1.0*inlier_src_lines.size()/src_lines.size()<<endl;
        
        vector<Eigen::Vector2d> inlier_src_pts;
        vector<Eigen::Vector2d> inlier_dst_pts;
        for (int i = 0; i<mask.size(); i++) {
            if (mask[i] != 0) {
                Eigen::Vector2d p(tracked_point_groups[0][i].x, tracked_point_groups[0][i].y);
                Eigen::Vector2d q(tracked_point_groups[1][i].x, tracked_point_groups[1][i].y);
                inlier_src_pts.push_back(p);
                inlier_dst_pts.push_back(q);
            }
        }
        //cout<<"inlier point number "<<inlier_src_pts.size()<<" . Percentage "<<1.0*inlier_src_pts.size()/mask.size()<<endl;
        
        Eigen::Matrix3d estimated_homo;
        bool is_estimated = VnlAlgoHomography::estimateHomography(inlier_src_pts, inlier_dst_pts,
                                                                  inlier_src_lines, inlier_dst_line_pts,
                                                                  h_eigen, estimated_homo);
        if (is_estimated) {
            eigen2cv(estimated_homo, h);
            inv_h = estimated_homo.inverse();
        }

        h.copyTo(output_warp);
        
        // step 5. analyze estimation quality
        if (warp_quality >= 0) {
            // inlier line space coverage
            Mat space = Mat::zeros(rows, cols, CV_8UC1);
            for (int i = 0; i<dst_centers.size(); i++) {
                // from dst point to source edge
                double dist = transformedPointLineDistance(inv_h, dst_centers[i][0], dst_centers[i][1],
                                                           src_lines[i][0], src_lines[i][1],
                                                           src_lines[i][2], src_lines[i][3]);
                // inlier
                if (dist < edge_to_line_reproj_threshold) {
                    cv::Rect r(dst_centers[i][0] - block_size/2, dst_centers[i][1] - block_size/2, block_size, block_size);
                    r.x = std::max(0, r.x);
                    r.y = std::max(0, r.y);
                    if (r.x + r.width >= cols) {
                        r.width = cols - r.x;
                    }
                    if (r.y + r.height >= rows) {
                        r.height = rows - r.y;
                    }
                    space(r) = 255;
                }
            }
            warp_quality = 1.0 * cv::countNonZero(space)/(rows * cols);           
            
            //EigenMatlabWriter::write_vector<double>(inlier_distance, "inlier.mat", "inlier");
            //EigenMatlabWriter::write_vector<double>(outlier_distance, "outlier.mat", "outlier");
            //cv::imshow("green inlier and red outlier", vis_im);
            //cv::waitKey();
        }
        
        return true;
    }
    
    
}


