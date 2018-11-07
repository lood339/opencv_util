//
//  cvx_model_tracking2d.cpp
//  CalibMeMatching
//
//  Created by jimmy on 2018-02-05.
//  Copyright (c) 2018 Nowhere Planet. All rights reserved.
//

#include "model_tracking2d.h"
#include "lsd_line_segment.h"
#include "imgproc.hpp"
#include "cvx_pgl_perspective_camera.h"
#include "cvx_draw.hpp"
#include "cvx_line.h"
#include <iostream>
#include "cvx_pgl_perspective_camera.h"

// for debug
#include "mat_io.hpp"
#include "cvx_draw.hpp"

using std::cout;
using std::endl;

namespace cvx {
    
    // detect lsd lines in image and cut them to short line segment
    static std::vector<cv::Vec4f> lsdLineDetection(const cv::Mat & im)
    {
        assert(im.type() == CV_8UC1);
        
        // to doulbe image
        Mat im_temp;
        im.convertTo(im_temp, CV_64FC1);
        if (!im_temp.isContinuous()) {
            im_temp = im_temp.clone();
        }
        
        const int cols = im.cols;
        const int rows = im.rows;
        std::vector<LSD::LSDLineSegment2D> lsd_lines;
        LSD::detectLines((double *) im_temp.data, cols, rows, lsd_lines);
        
        LSD::shortenLineSegments(lsd_lines, 25);    // divide long lines to short lines
        vector<cv::Vec4f> ret_lines(lsd_lines.size());
        for (int i = 0; i<lsd_lines.size(); i++) {
            ret_lines[i] = cv::Vec4f(lsd_lines[i].x1, lsd_lines[i].y1, lsd_lines[i].x2, lsd_lines[i].y2);
        }
        return ret_lines;
    }

    // divide line segment into short line segment by half of the line segment
    static vector<std::pair<Eigen::Vector2d, Eigen::Vector2d> > getHalfSegment(const std::pair<Eigen::Vector2d, Eigen::Vector2d>& line, const double max_length)
    {
        const Eigen::Vector2d p1 = line.first;
        const Eigen::Vector2d p2 = line.second;
        
        using LineType = std::pair<Eigen::Vector2d, Eigen::Vector2d>;
        double dist = (p1 - p2).norm();
        if (dist <= max_length) {
            vector<std::pair<Eigen::Vector2d, Eigen::Vector2d> > lines;
            lines.push_back(line);
            return lines;
        }
        else {
            Eigen::Vector2d mid_p = (p1 + p2)/2.0;
            LineType line1 = line;
            line1.second = mid_p;
            LineType line2 = line;
            line2.first = mid_p;
            
            vector<LineType> left_lines = getHalfSegment(line1, max_length);
            vector<LineType> right_lines = getHalfSegment(line2, max_length);
            assert(left_lines.size() >= 1 && right_lines.size() >= 1);
            
            left_lines.insert(left_lines.end(), right_lines.begin(), right_lines.end());
            return left_lines;
        }
    }
    
    // project line segment into image
    // return: line segment that inside the image
    static void projectLineSegment(const cvx_pgl::perspective_camera& camera,
                                   const vector<std::pair<Eigen::Vector2d, Eigen::Vector2d> > & line_segments,
                                   const int im_w, const int im_h,
                                   vector<std::pair<Eigen::Vector2d, Eigen::Vector2d> > & projected_line_segments)
    {
        using LineType = std::pair<Eigen::Vector2d, Eigen::Vector2d>;
       
        Eigen::AlignedBox<double, 2> im_rect(Vector2d(0, 0), Vector2d(im_w, im_h));
        for (int i = 0; i<line_segments.size(); i++) {
            Eigen::Vector2d q1 = camera.project2d(line_segments[i].first);
            Eigen::Vector2d q2 = camera.project2d(line_segments[i].second);
            
            if (im_rect.contains(q1) && im_rect.contains(q2)) {
                projected_line_segments.push_back(LineType(q1, q2));
            }
        }
    }
    
    // interface for function cvx::trackLineSegmentCenter()
    static void trackLineSegmentCenter(const vector<std::pair<Eigen::Vector2d, Eigen::Vector2d> >& _src,
                                       const vector<std::pair<Eigen::Vector2d, Eigen::Vector2d> >& _dst,
                                       vector<Eigen::Vector2d>& _centers,
                                       const cv::Size& image_size,
                                       cv::OutputArray center_point_patch_distance = cv::noArray(),
                                       cv::InputOutputArray dist_map = cv::noArray(),
                                       double search_length = 10,
                                       int block_size = 25,
                                       bool given_dist_map = false)
    {
        vector<cv::Vec4f> src;
        vector<cv::Vec4f> dst;
        for (int i = 0; i<_src.size(); i++) {
            Eigen::Vector2d p1 = _src[i].first;
            Eigen::Vector2d p2 = _src[i].second;
            cv::Vec4f e(p1.x(), p1.y(), p2.x(), p2.y());
            src.push_back(e);
        }
        
        for (int i = 0; i<_dst.size(); i++) {
            Eigen::Vector2d p1 = _dst[i].first;
            Eigen::Vector2d p2 = _dst[i].second;
            cv::Vec4f e(p1.x(), p1.y(), p2.x(), p2.y());
            dst.push_back(e);
        }
        
        vector<cv::Vec2f> centers;
        cvx::trackLineSegmentCenter(src, dst, centers, image_size,
                                    center_point_patch_distance, dist_map,
                                    search_length,
                                    block_size,
                                    given_dist_map);
        for (int i = 0; i<centers.size(); i++) {
            _centers.push_back(Eigen::Vector2d(centers[i][0], centers[i][1]));
        }
        assert(_centers.size() == _src.size());
    }
    
    namespace {
        struct Line2D {
            std::pair<Vector2d, Vector2d> world_line;
            Eigen::ParametrizedLine<double, 2> image_line;
            std::vector<Vector2d> image_points;  // inlier support point
        };
    }
    

    
bool trackEdgeImage(const cvx_pgl::perspective_camera& init_camera,
                    const vector<std::pair<Vector2d, Vector2d> >& _lines,
                    const cv::Mat& _edge_image,
                    cvx_pgl::perspective_camera& refined_camera)
{
    assert(_lines.size() > 1);
    assert(_edge_image.type() == CV_8UC1);    
    
    refined_camera = init_camera;
    const int im_w = _edge_image.cols;
    const int im_h = _edge_image.rows;
    cv::Size im_size(im_w, im_h);
    const double search_length = 20;
    const int block_size = 25;
    const double point_to_line_inlier_threshold = 2.0;
    
    using LineType = std::pair<Eigen::Vector2d, Eigen::Vector2d>;
    
    // step 1: detect edge in edge map and compute distance map
    vector<cv::Vec4f> detected_edges = lsdLineDetection(_edge_image);
    vector<cv::Vec4f> dummy_edges;
    vector<cv::Vec2f> dummy_centers;
    cv::Mat distance_map;
    cvx::trackLineSegmentCenter(dummy_edges, detected_edges, dummy_centers,
                                im_size, cv::noArray(),
                                distance_map,
                                search_length, block_size, false);
    assert(distance_map.type() == CV_32FC1);
    
    cv::Mat debug_im;
    _edge_image.copyTo(debug_im);
    cv::cvtColor(debug_im, debug_im, CV_GRAY2BGR);
    
    // step 2: track edge center and estimate tracked lines
    vector<Line2D> tracked_lines;  // tracking result
    for (int i = 0; i<_lines.size(); i++) {        
        vector<LineType> short_lines = getHalfSegment(_lines[i], 1.0); // one line segment for each meter
        
        // project short lines
        vector<LineType> projected_short_lines;
        projectLineSegment(init_camera, short_lines, im_w, im_h, projected_short_lines);
        
        // too few line segment inside image
        if (projected_short_lines.size() < 5) {
            continue;
        }
        printf("projected short line number is %lu\n", projected_short_lines.size());
        
        // track short lines
        vector<Eigen::Vector2d> tracked_edge_centers;
        vector<LineType> dummy_lines;
        assert(distance_map.type() == CV_32FC1);
        trackLineSegmentCenter(projected_short_lines,
                               dummy_lines,
                               tracked_edge_centers,
                               im_size, cv::noArray(),
                               distance_map, search_length, block_size, true);
        
        /*
         // visualize line in the edge map
        
         */
        
        // estimate line with ransac
        Eigen::ParametrizedLine<double, 2> estimated_line;
        vector<int> inlier_index;
        bool is_fit = fitLine2DRansac(tracked_edge_centers, estimated_line, inlier_index, point_to_line_inlier_threshold);
        if (!is_fit) {
            continue;
        }
        
        Line2D tracked_line;
        tracked_line.world_line = _lines[i];
        tracked_line.image_line = estimated_line;
        for (int j = 0; j<inlier_index.size(); j++) {
            tracked_line.image_points.push_back(tracked_edge_centers[j]);
        }
        
        tracked_lines.push_back(tracked_line);
    }
   
    if (tracked_lines.size() < 4) {
        printf("tracked lines number is %lu, less than 4.\n", tracked_lines.size());
        return false;
    }
    
    // step 3: data association in tracked lines
    Eigen::AlignedBox<double, 2> im_rect(Vector2d(0, 0), Vector2d(im_w, im_h));
    vector<Vector2d> model_pts;
    vector<Vector2d> im_pts;
    vector<std::pair<Eigen::Vector2d, Eigen::Vector2d> > model_lines;
    vector<Vector2d> im_line_pts;
    Eigen::MatrixXd model_im_points(9, 4);
    int num = 0;
    for (int i = 0; i<tracked_lines.size(); i++) {
        for (int j = i+1; j<tracked_lines.size(); j++) {
            Line2D line1 = tracked_lines[i];
            Line2D line2 = tracked_lines[j];
            Eigen::Vector2d dir1 = line1.world_line.first - line1.world_line.second;
            dir1.normalize();
            Eigen::Vector2d dir2 = line2.world_line.first - line2.world_line.second;
            dir2.normalize();
            double cos_angle = fabs(dir1.dot(dir2)); // parallel line will have large absolute value
            if (cos_angle > 0.5) {
                continue;                // parall lines
            }
            
            // intersection of two world line
            Eigen::ParametrizedLine<double, 2> world_line1 = Eigen::ParametrizedLine<double, 2>::Through(line1.world_line.first,
                                                                                                         line1.world_line.second);
            Eigen::Hyperplane<double, 2> world_line2 = Eigen::Hyperplane<double, 2>(line2.world_line.first,
                                                                                    line2.world_line.second);
            Eigen::Vector2d world_intersection = world_line1.intersectionPoint(world_line2);
            Eigen::Vector2d image_intersection = line1.image_line.intersectionPoint(Eigen::Hyperplane<double, 2>(line2.image_line));
            
            
            Eigen::Vector2d q = init_camera.project2d(world_intersection);
            double dis = (image_intersection-q).norm();
            if (dis > 50) {
                continue;
            }
            
            if (im_rect.contains(image_intersection)) {
                model_pts.push_back(world_intersection);
                im_pts.push_back(image_intersection);
                model_im_points(num, 0) = world_intersection.x();
                model_im_points(num, 1) = world_intersection.y();
                model_im_points(num, 2) = image_intersection.x();
                model_im_points(num, 3) = image_intersection.y();
                num++;
            }
        }
    }
    printf("valid intersection num is %lu\n", model_pts.size());
    
    if (model_pts.size() < 4) {
        return false;
    }
    
    // reprojection error by initial camera
    for (int i = 0; i<model_pts.size(); i++) {
        Eigen::Vector2d p = im_pts[i];
        Eigen::Vector2d q = init_camera.project2d(model_pts[i]);
        double dis = (p-q).norm();
        printf("%d\t %lf\n", i, dis);
    }
    
    
    /*
    vector<cv::Point2f> debug_points;
    for (int j = 0; j<im_pts.size(); j++) {
        debug_points.push_back(cv::Point2f(im_pts[j].x(), im_pts[j].y()));
    }
    CvxDraw::draw_cross_template<cv::Point2f>(debug_im, debug_points, CvxDraw::green(), 10, 2);
    cv::imshow("estimated line intersection", debug_im);
    cv::waitKey();
     */
    /*
    matio::writeMatrix("6_data_associate.mat", "world_img_pts", model_im_points);
    
    vector<cv::Point2f> debug_points;
    for (int j = 0; j<model_pts.size(); j++) {
        debug_points.push_back(cv::Point2f(model_pts[j].x()*10, model_pts[j].y()*10));
    }
    CvxDraw::draw_cross_template<cv::Point2f>(debug_im, debug_points, CvxDraw::green(), 10, 2);
    cv::imshow("estimated line intersection", debug_im);
    cv::waitKey();
     */
    
    double reproj_error = cvx_pgl::estimateCamera(model_pts, im_pts, model_lines, im_line_pts, init_camera, refined_camera);
    printf("reporjection error is %lf\n", reproj_error);
    return true;
}
    
    
    /*****************    edgeSearch spoort function    ***************/
    // detect lsd lines in image and cut them to short line segment
    // @brief mask if one pixel is on a line. 0 or 255
    static void lsdLineMask(const cv::Mat & im, cv::Mat& mask)
    {
        assert(im.type() == CV_8UC1);
        
        // step 1: detect lines
        // to doulbe image
        Mat im_temp;
        im.convertTo(im_temp, CV_64FC1);
        if (!im_temp.isContinuous()) {
            im_temp = im_temp.clone();
        }
        const int cols = im.cols;
        const int rows = im.rows;
        std::vector<LSD::LSDLineSegment2D> lsd_lines;
        LSD::detectLines((double *) im_temp.data, cols, rows, lsd_lines);
        
        // step 2: draw lines in a mask image
        mask = cv::Mat::zeros(im.rows, im.cols, CV_8UC1);
        for (int i = 0; i<lsd_lines.size(); i++) {
            int x1 = cvRound(lsd_lines[i].x1);
            int y1 = cvRound(lsd_lines[i].y1);
            int x2 = cvRound(lsd_lines[i].x2);
            int y2 = cvRound(lsd_lines[i].y2);
            int line_width = cvRound(lsd_lines[i].width_);
            cv::line(mask, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar::all(255), line_width);
        }
    }
    
        
    
    bool edgeSearch(const cvx_pgl::perspective_camera& _init_camera,
                    const vector<Vector2d>& _model_points,
                    const vector<Vector2d>& _model_point_normal_direction,
                    const cv::Mat& _image,
                    const double _search_distance,
                    cvx_pgl::perspective_camera& refined_camera)
    {
        // check input
        assert(_model_points.size() > 1);
        assert(_image.type() == CV_8UC1 || _image.type() == CV_8UC3);
        assert(_model_points.size() == _model_point_normal_direction.size());
        
        
        Mat gray_image;
        if (_image.channels() == 3) {
            cvtColor( _image, gray_image, CV_BGR2GRAY );
        }
        else {
            gray_image = _image;
        }

        refined_camera = _init_camera;
        const int im_w = _image.cols;
        const int im_h = _image.rows;
        cv::Size im_size(im_w, im_h);
        const cv::Size sz(im_w, im_h);
        const double search_length = _search_distance/2.0;
        
        // support data structure
        Mat bw = Mat(im_h, im_w, CV_8UC1, cv::Scalar(255));
        
        // step 1: detect edge in edge map and compute distance map
        Mat grad_x, grad_y, grad_mag;
        cvx::imgproc::gradient(gray_image, grad_x, grad_y, grad_mag);
        assert(grad_mag.type() == CV_64FC1);
        assert(grad_x.type() == CV_64FC1);
        assert(grad_y.type() == CV_64FC1);
        
        Mat line_mask;
        lsdLineMask(gray_image, line_mask);
        assert(line_mask.type() == CV_8UC1);
        
        //cv::imshow("line mask", line_mask);
        //cv::waitKey();
        
        // step 2: projected points and its normal direction
        // the search is with line detection and distance transform
        vector<Eigen::Vector2d> observed_model_points;  // world coordinate
        vector<Eigen::Vector2d> searched_points;        // image coordinate
        Eigen::AlignedBox<double, 2> im_rect(Vector2d(0, 0), Vector2d(im_w, im_h));
        const double angular_threahold = cos(45.0/180.0*M_PI);    // smaller than 20 degrees
        for (int i = 0; i<_model_points.size(); i++) {
            Vector2d p = _model_points[i];
            Vector2d n = _model_point_normal_direction[i];
            Vector2d p1 = p + search_length * _model_point_normal_direction[i];
            Vector2d p2 = p - search_length * _model_point_normal_direction[i];
            
            Eigen::Vector2d q = _init_camera.project2d(p);
            Eigen::Vector2d q1 = _init_camera.project2d(p1);
            Eigen::Vector2d q2 = _init_camera.project2d(p2);
            Vector2d img_dir = q1 - q2;
            img_dir = img_dir.normalized();     // projected point direction           
        
          
            
            // three point must be inside the image
            if (!im_rect.contains(q) || !im_rect.contains(q1) || !im_rect.contains(q2)) {
                continue;
            }
            
            // search the location, 1. with maximum gradient, 2. similar orentation
            
            // [q1 q2] a virtual line along the normal direction
            cv::Point q1_img(cvRound(q1.x()), cvRound(q1.y()));
            cv::Point q2_img(cvRound(q2.x()), cvRound(q2.y()));
            cv::clipLine(sz, q1_img, q2_img);
            cv::LineIterator it(bw, q1_img, q2_img, 8);
            
            double max_mag = -1.0;
            cv::Point max_p(p.x(), p.y());
            for(int ct = 0; ct < it.count; ct++, ++it)
            {
                cv::Point p = it.pos();  // center location of the patch
                int r = p.y;
                int c = p.x;
                double mag = grad_mag.at<double>(r, c);
                // on line segment and with large magnitude
                if (line_mask.at<unsigned char>(r, c) == 255 && mag > max_mag) {
                    
                    double dx = grad_x.at<double>(r, c);
                    double dy = grad_y.at<double>(r, c);
                    Eigen::Vector2d grad_dir(dx, dy);
                    grad_dir = grad_dir.normalized();
                    
                    // similar direction
                    double angular_dif = std::abs(img_dir.dot(grad_dir));
                    if (angular_dif > angular_threahold) {
                        max_mag = mag;
                        max_p = p;
                    }
                }
            }
            if (max_mag > 0) {
                observed_model_points.push_back(_model_points[i]);
                searched_points.push_back(Eigen::Vector2d(max_p.x, max_p.y));
            }
        }
        
        if (1)
        {
            cv::Mat debug_image;
            _image.copyTo(debug_image);
            // debug
            vector<cv::Point2d> projected_pts;
            for (int i = 0; i<searched_points.size(); i++) {
                Eigen::Vector2d p = searched_points[i];
                projected_pts.push_back(cv::Point2d(p.x(), p.y()));
            }
            
            CvxDraw::draw_cross_template(debug_image, projected_pts, CvxDraw::green());
            cv::imshow("searched points", debug_image);
            cv::waitKey();
        }
        
        assert(observed_model_points.size() == searched_points.size());
        
        if (observed_model_points.size() < 8) {
            return false;
        }
        
        // step 3: RANSAC to optimize camera pose
        bool is_refined = cvx_pgl::refineCameraRANSAC(observed_model_points, searched_points,
                                                      _init_camera, refined_camera);
        return is_refined;
    }
    
}