//
//  cvx_pl_pose_estimation.cpp
//  PointLineReloc
//
//  Created by jimmy on 2017-04-06.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "cvx_pl_pose_estimation.h"
#include "cvxCalib3d.hpp"
#include <algorithm>
#include "vnl_algo.h"
#include <string>
#include <unordered_map>
#include "vpgl_algo.h"
#include <opencv2/core/eigen.hpp>


using std::unordered_map;
using std::string;


bool PreemptiveRANSAC3DPointLineParameter::readFromFile(const char *file)
{
    FILE *pf = fopen(file, "r");
    assert(pf);
    const int param_num = 4;
    std::unordered_map<std::string, double> imap;
    for(int i = 0; i<param_num; i++)
    {
        char s[1024] = {NULL};
        double val = 0.0;
        int ret = fscanf(pf, "%s %lf", s, &val);
        if (ret != 2) {
            break;
        }
        imap[string(s)] = val;
    }
    assert(imap.size() == param_num);
    
    dis_threshold = imap[string("dis_threshold")];
    line_distance_threshold = imap[string("line_distance_threshold")];
    line_loss_weight = imap[string("line_loss_weight")];
    non_linear_optimization = ((int)imap[string("non_linear_optimization")] != 0);
    fclose(pf);
    return true;
}


struct CameraPoseHypothese
{
    double loss_;
    Mat rvec_;       // rotation     for 3D --> 2D projection
    Mat tvec_;       // translation  for 3D --> 2D projection
    Mat affine_;     //              for 3D --> 3D camera to world transformation
    vector<unsigned int> inlier_indices_;         // camera coordinate index
    vector<unsigned int> inlier_candidate_world_pts_indices_; // candidate world point index
    vector<unsigned int> line_indices_;   // inlier line
    
    // store all inliers from preemptive ransac, for line
    vector<cv::Point3d> camera_pts_;
    vector<cv::Point3d> wld_pts_;
    
    CameraPoseHypothese()
    {
        loss_ = INT_MAX;
    }
    CameraPoseHypothese(const double loss)
    {
        loss_  = loss;
    }
    
    CameraPoseHypothese(const CameraPoseHypothese & other)
    {
        loss_ = other.loss_;
        rvec_ = other.rvec_;
        tvec_ = other.tvec_;
        affine_ = other.affine_;
        inlier_indices_.clear();
        inlier_indices_.resize(other.inlier_indices_.size());
        inlier_candidate_world_pts_indices_.clear();
        inlier_candidate_world_pts_indices_.resize(other.inlier_candidate_world_pts_indices_.size());       
        for(int i = 0; i<other.inlier_indices_.size(); i++) {
            inlier_indices_[i] = other.inlier_indices_[i];
        }
        for(int i = 0; i<other.inlier_candidate_world_pts_indices_.size(); i++){
            inlier_candidate_world_pts_indices_[i] = other.inlier_candidate_world_pts_indices_[i];
        }
        if(inlier_candidate_world_pts_indices_.size() != 0){
            assert(inlier_indices_.size() == inlier_candidate_world_pts_indices_.size());
        }
        
        // copy camera points and world coordinate points
        assert(other.camera_pts_.size() == other.wld_pts_.size());
        camera_pts_.resize(other.camera_pts_.size());
        wld_pts_.resize(other.wld_pts_.size());
        line_indices_.resize(other.line_indices_.size());
        std::copy(other.camera_pts_.begin(), other.camera_pts_.end(), camera_pts_.begin());
        std::copy(other.wld_pts_.begin(), other.wld_pts_.end(), wld_pts_.begin());
        std::copy(other.line_indices_.begin(), other.line_indices_.end(), line_indices_.begin());
    }
    
    bool operator < (const CameraPoseHypothese & other) const
    {
        return loss_ < other.loss_;
    }
    
    CameraPoseHypothese & operator = (const CameraPoseHypothese & other)
    {
        if (&other == this) {
            return *this;
        }
        loss_ = other.loss_;
        rvec_ = other.rvec_;
        tvec_ = other.tvec_;
        affine_ = other.affine_;
        inlier_indices_.clear();
        inlier_indices_.resize(other.inlier_indices_.size());
        inlier_candidate_world_pts_indices_.clear();
        inlier_candidate_world_pts_indices_.resize(other.inlier_candidate_world_pts_indices_.size());
        
        for(int i = 0; i<other.inlier_indices_.size(); i++) {
            inlier_indices_[i] = other.inlier_indices_[i];
        }
        for(int i = 0; i<other.inlier_candidate_world_pts_indices_.size(); i++){
            inlier_candidate_world_pts_indices_[i] = other.inlier_candidate_world_pts_indices_[i];
        }
        if(inlier_candidate_world_pts_indices_.size() != 0){
            assert(inlier_indices_.size() == inlier_candidate_world_pts_indices_.size());
        }
        
        // copy camera points and world coordinate points
        // copy camera points and world coordinate points
        assert(other.camera_pts_.size() == other.wld_pts_.size());
        camera_pts_.resize(other.camera_pts_.size());
        wld_pts_.resize(other.wld_pts_.size());
        line_indices_.resize(other.line_indices_.size());
        std::copy(other.camera_pts_.begin(), other.camera_pts_.end(), camera_pts_.begin());
        std::copy(other.wld_pts_.begin(), other.wld_pts_.end(), wld_pts_.begin());
        std::copy(other.line_indices_.begin(), other.line_indices_.end(), line_indices_.begin());
        return *this;
    }
};


// projected_pt: projected point location on line
// return distance between the point "pt" and project point, or the point to the line distance
static double project3DPoint(const cv::Point3d& pt,
                             const Eigen::ParametrizedLine<double, 3>& line,
                             cv::Point3d& projected_pt)
{
    double dist = 0.0;
    Eigen::Vector3d p(pt.x, pt.y, pt.z);
    Eigen::Vector3d proj_p = line.projection(p);
    projected_pt.x = proj_p.x();
    projected_pt.y = proj_p.y();
    projected_pt.z = proj_p.z();
    dist = (p - proj_p).norm();
    
    return dist;
}

static vector<Eigen::Vector3d> transformPointList(const vector<cv::Point3d> & pts)
{
    vector<Eigen::Vector3d> pts2(pts.size());
    for (int i = 0; i<pts.size(); i++) {
        double x = pts[i].x;
        double y = pts[i].y;
        double z = pts[i].z;
        pts2[i] = Eigen::Vector3d(x, y, z);
    }
    return pts2;
}

static Eigen::Affine3d transformAffine(const Mat & affine)
{
    Eigen::Affine3d ret;
    for (int r = 0; r<3; r++) {
        for (int c = 0; c<4; c++) {
            ret(r, c) = affine.at<double>(r, c);
        }
    }
    return ret;
}

static Mat transformAffine(const Eigen::Affine3d & affine)
{
    Mat ret = cv::Mat::zeros(3, 4, CV_64FC1);
    for (int i = 0; i<3; i++) {
        for (int j = 0; j<4; j++) {
            ret.at<double>(i, j) = affine(i, j);
        }
    }
    return ret;
}

bool CvxPLPoseEstimation::preemptiveRANSAC3DOneToMany(const vector<cv::Point3d> & camera_pts,
                                                      const vector<vector<cv::Point3d> > & candidate_wld_pts,
                                                      const vector<cv::Point3d> & line_start_points,
                                                      const vector<cv::Point3d> & line_end_points,
                                                      const vector<Eigen::ParametrizedLine<double, 3> > & candidate_wld_lines,
                                                      const PreemptiveRANSAC3DPointLineParameter & param,
                                                      cv::Mat & camera_pose)
{
    assert(camera_pts.size() == candidate_wld_pts.size());
    assert(line_start_points.size() == line_end_points.size());
    assert(candidate_wld_lines.size() == line_start_points.size());
    
    const int num_iteration = 2048;
    const int K = 1024;
    const int N = (int)camera_pts.size();
    const int B = 500;
    
    // initial camera pose
    vector<cv::Mat > affine_candidate;
    for (int i = 0; i<num_iteration; i++) {
        int k1 = 0, k2 = 0, k3 = 0, k4 = 0;
        do{
            k1 = rand()%N;
            k2 = rand()%N;
            k3 = rand()%N;
            k4 = rand()%N;
        }while (k1 == k2 || k1 == k3 || k1 == k4 ||
                k2 == k3 || k2 == k4 || k3 == k4);
        
        vector<cv::Point3d> sampled_camera_pts;
        vector<cv::Point3d> sampled_wld_pts;
        
        sampled_camera_pts.push_back(camera_pts[k1]);
        sampled_camera_pts.push_back(camera_pts[k2]);
        sampled_camera_pts.push_back(camera_pts[k3]);
        sampled_camera_pts.push_back(camera_pts[k4]);
        
        sampled_wld_pts.push_back(candidate_wld_pts[k1][0]);
        sampled_wld_pts.push_back(candidate_wld_pts[k2][0]);
        sampled_wld_pts.push_back(candidate_wld_pts[k3][0]);
        sampled_wld_pts.push_back(candidate_wld_pts[k4][0]);
        
        Mat affine;
        CvxCalib3D::KabschTransform(sampled_camera_pts, sampled_wld_pts, affine);
        affine_candidate.push_back(affine);
        if (affine_candidate.size() > K) {
            printf("initialization repeat %d times\n", i);
            break;
        }
    }
    printf("number of candidate camera pose is %lu\n", affine_candidate.size());
    
    vector<CameraPoseHypothese> losses;
    for (int i = 0; i<affine_candidate.size(); i++) {
        CameraPoseHypothese hyp(0.0);
        hyp.affine_ = affine_candidate[i];
        losses.push_back(hyp);
    }
    
    const double threshold = param.dis_threshold;
    const double line_threshold = param.line_distance_threshold;
    const double line_loss_weight = param.line_loss_weight;
    const bool non_linear_opt = param.non_linear_optimization;
    while (losses.size() > 1) {
        // sample random set
        vector<cv::Point3d> sampled_camera_pts;
        vector< vector<cv::Point3d> > sampled_wld_pts;  // one camera point may have multiple world points correspondences
        vector<int> sampled_indices;
        for (int i = 0; i<B; i++) {
            int index = rand()%N;
            sampled_camera_pts.push_back(camera_pts[index]);
            sampled_wld_pts.push_back(candidate_wld_pts[index]);
            sampled_indices.push_back(index);
        }
        
        // count outliers
        for (int i = 0; i<losses.size(); i++) {
            // evaluate the accuracy by check transformation
            vector<cv::Point3d> transformed_pts;
            CvxCalib3D::rigidTransform(sampled_camera_pts, losses[i].affine_, transformed_pts);
            
            // check minimum distance from transformed points to world coordiante
            for (int j = 0; j<transformed_pts.size(); j++) {
                double min_dis = threshold * 2;
                int min_index = -1;
                for (int k = 0; k<sampled_wld_pts[j].size(); k++) {
                    cv::Point3d dif = transformed_pts[j] - sampled_wld_pts[j][k];
                    double dis = cv::norm(dif);
                    if (dis < min_dis) {
                        min_dis = dis;
                        min_index = k;
                    }
                } // end of k
                
                if (min_dis > threshold) {
                    losses[i].loss_ += 1.0;
                }
                else {
                    losses[i].inlier_indices_.push_back(sampled_indices[j]);
                    losses[i].inlier_candidate_world_pts_indices_.push_back(min_index);
                }
            } // end of j
            assert(losses[i].inlier_indices_.size() == losses[i].inlier_candidate_world_pts_indices_.size());
            // printf("inlier number is %lu\n", losses[i].inlier_indices_.size());
            
            // check lines
            vector<cv::Point3d> transformed_start_pts;  // world coordinate
            vector<cv::Point3d> transformed_end_pts;
            CvxCalib3D::rigidTransform(line_start_points, losses[i].affine_, transformed_start_pts);
            CvxCalib3D::rigidTransform(line_end_points, losses[i].affine_, transformed_end_pts);
            for (int j = 0; j<candidate_wld_lines.size(); j++) {  // line index
                cv::Point3d proj_start_pt;
                cv::Point3d proj_end_pt;
                double dist1 = project3DPoint(transformed_start_pts[j], candidate_wld_lines[j], proj_start_pt);
                double dist2 = project3DPoint(transformed_end_pts[j], candidate_wld_lines[j], proj_end_pt);
                if (dist1 < line_threshold && dist2 < line_threshold ) {
                    losses[i].camera_pts_.push_back(line_start_points[j]);
                    losses[i].camera_pts_.push_back(line_end_points[j]);
                    losses[i].wld_pts_.push_back(proj_start_pt);
                    losses[i].wld_pts_.push_back(proj_end_pt);
                    losses[i].line_indices_.push_back(j);
                }
                else {
                    losses[i].loss_ += line_loss_weight;  // 
                }
            } // end of j
        }
        
        std::sort(losses.begin(), losses.end());
        losses.resize(losses.size()/2);
        
        for (int i = 0; i<losses.size(); i++) {
            //   printf("after: loss is %lf\n", losses[i].loss_);
            //   printf("inlier number is %lu\n", losses[i].inlier_indices_.size());
        }
        // printf("\n\n");
        
        // refine by inliers
        for (int i = 0; i<losses.size(); i++) {
            // number of inliers is larger than minimum configure
            if (losses[i].inlier_indices_.size() > 4) {
                vector<cv::Point3d> inlier_camera_pts;
                vector<cv::Point3d> inlier_wld_pts;
                for (int j = 0; j < losses[i].inlier_indices_.size(); j++) {
                    int index = losses[i].inlier_indices_[j];
                    int wld_index = losses[i].inlier_candidate_world_pts_indices_[j];
                    inlier_camera_pts.push_back(camera_pts[index]);
                    inlier_wld_pts.push_back(candidate_wld_pts[index][wld_index]);
                }
                
                Mat affine;
                // inlier line segment
                if (losses[i].camera_pts_.size() != 0) {
                    CvxCalib3D::KabschTransform(inlier_camera_pts, inlier_wld_pts, losses[i].camera_pts_, losses[i].wld_pts_, affine);
                    
                    // the final camera output
                    if (losses.size() == 1 && non_linear_opt) {
                        assert(losses[i].line_indices_.size() * 2 == losses[i].camera_pts_.size());
                        
                        vector<Eigen::Vector3d> camera_start_pts;
                        vector<Eigen::Vector3d> camera_end_pts;
                        vector<Eigen::ParametrizedLine<double, 3> > world_lines;
                        for (int j = 0; j<losses[i].line_indices_.size(); j++) {
                            int index = losses[i].line_indices_[j];
                            world_lines.push_back(candidate_wld_lines[index]);
                            
                            cv::Point3d p1 = losses[i].camera_pts_[2 * j];
                            cv::Point3d p2 = losses[i].camera_pts_[2 * j + 1];
                            Eigen::Vector3d p3(p1.x, p1.y, p1.z);
                            Eigen::Vector3d p4(p2.x, p2.y, p2.z);
                            camera_start_pts.push_back(p3);
                            camera_end_pts.push_back(p4);
                        }
                        
                        Eigen::Affine3d init_camera = transformAffine(affine);
                        Eigen::Affine3d final_camera;
                        bool is_estimated = VnlAlgo::estimateCameraPose(transformPointList(inlier_camera_pts),
                                                                        transformPointList(inlier_wld_pts),
                                                                        camera_start_pts, camera_end_pts,
                                                                        world_lines, init_camera, final_camera);
                        if (is_estimated) {
                            affine = transformAffine(final_camera);
                        }
                    }
                }
                else {
                     CvxCalib3D::KabschTransform(inlier_camera_pts, inlier_wld_pts, affine);                    
                }                
               
                losses[i].affine_ = affine;
                losses[i].inlier_indices_.clear();
                losses[i].inlier_candidate_world_pts_indices_.clear();
                losses[i].camera_pts_.clear();
                losses[i].wld_pts_.clear();
                losses[i].line_indices_.clear();
            }
        }
    }
    assert(losses.size() == 1);
    
    camera_pose = cv::Mat::eye(4, 4, CV_64F);
    losses[0].affine_.copyTo(camera_pose(cv::Rect(0, 0, 4, 3)));
    return true;
}

static Eigen::Affine3d getProjectionMatrix(const cv::Mat& camera_matrix,
                                           const cv::Mat& rvec,
                                           const cv::Mat& tvec)
{
    assert(rvec.type() == CV_64FC1);
    assert(tvec.type() == CV_64FC1);
    
    Eigen::Affine3d affine;
    Mat rot;
    cv::Rodrigues(rvec, rot);
    for (int r = 0; r<3; r++) {
        for (int c = 0; c<3; c++) {
            affine(r, c) = rot.at<double>(r, c);
        }
    }
    affine(0, 3) = tvec.at<double>(0, 0);
    affine(1, 3) = tvec.at<double>(1, 0);
    affine(2, 3) = tvec.at<double>(2, 0);
    
    Eigen::Matrix3d eigen_camera_matrix;
    cv2eigen(camera_matrix, eigen_camera_matrix);
    
    affine = eigen_camera_matrix * affine;
    return affine;
}

static Eigen::Affine3d getAffineMatrix(const cv::Mat& rvec,
                                       const cv::Mat& tvec)
{
    assert(rvec.type() == CV_64FC1);
    assert(tvec.type() == CV_64FC1);
    
    Eigen::Affine3d affine;
    Mat rot;
    cv::Rodrigues(rvec, rot);
    for (int r = 0; r<3; r++) {
        for (int c = 0; c<3; c++) {
            affine(r, c) = rot.at<double>(r, c);
        }
    }
    affine(0, 3) = tvec.at<double>(0, 0);
    affine(1, 3) = tvec.at<double>(1, 0);
    affine(2, 3) = tvec.at<double>(2, 0);
    return affine;
}

static void affineToRotationTranslation(const Eigen::Affine3d & affine, cv::Mat& rvec, cv::Mat& tvec)
{
    Eigen::Matrix3d R  = affine.linear();
    cv::Mat r_mat = cv::Mat::zeros(3, 3, CV_64FC1);
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            r_mat.at<double>(r, c) = R(r, c);
        }
    }
    
    cv::Rodrigues(r_mat, rvec);
    Eigen::Vector3d translation = affine.translation();
    tvec = cv::Mat::zeros(3, 1, CV_64FC1);
    tvec.at<double>(0, 0) = translation[0];
    tvec.at<double>(1, 0) = translation[1];
    tvec.at<double>(2, 0) = translation[2];
}


static Eigen::Vector2d cv2eigen(const cv::Point2d& cv_pt)
{
    Eigen::Vector2d p(cv_pt.x, cv_pt.y);
    return p;
}

static void cv2eigen(const vector<cv::Point2d> & cv_pt, vector<Eigen::Vector2d> & eigen_pt)
{
    eigen_pt.resize(cv_pt.size());
    for (int i = 0; i<cv_pt.size(); i++) {
        eigen_pt[i] = Eigen::Vector2d(cv_pt[i].x, cv_pt[i].y);
    }
}

static void cv2eigen(const vector<cv::Point3d> & cv_pt, vector<Eigen::Vector3d> & eigen_pt)
{
    eigen_pt.resize(cv_pt.size());
    for (int i = 0; i<cv_pt.size(); i++) {
        eigen_pt[i] = Eigen::Vector3d(cv_pt[i].x, cv_pt[i].y, cv_pt[i].z);
    }
}

bool CvxPLPoseEstimation::preemptiveRANSAC2DOneToMany(const vector<cv::Point2d> & img_pts,
                                                      const vector<vector<cv::Point3d> > & candidate_wld_pts,
                                                      const vector<cv::Point2d> & line_start_points,
                                                      const vector<cv::Point2d> & line_end_points,
                                                      const vector<Eigen::ParametrizedLine<double, 3> > & candidate_wld_lines,
                                                      const cv::Mat & camera_matrix,
                                                      const cv::Mat & dist_coeff,
                                                      const PreemptiveRANSAC2DParameter & param,
                                                      cv::Mat & camera_pose)
{
    printf("warning: CvxPLPoseEstimation::preemptiveRANSAC2DOneToMany is not stable.\n");
    assert(img_pts.size() == candidate_wld_pts.size());
    assert(line_end_points.size() == line_start_points.size());
    assert(line_end_points.size() == candidate_wld_lines.size());
    
    const int num_iteration = 2048;
    int K = 1024;
    const int N = (int)img_pts.size();
    int B = 500;
    if (img_pts.size() < 1000) {
        B = 300;
    }
    
    vector<std::pair<Mat, Mat> > rt_candidate;
    for (int i = 0; i<num_iteration; i++) {
        int k1 = 0, k2 = 0, k3 = 0, k4 = 0;
        do{
            k1 = rand()%N;
            k2 = rand()%N;
            k3 = rand()%N;
            k4 = rand()%N;
        }while (k1 == k2 || k1 == k3 || k1 == k4 ||
                k2 == k3 || k2 == k4 || k3 == k4);
        
        vector<cv::Point2d> sampled_img_pts;
        vector<cv::Point3d> sampled_wld_pts;
        
        sampled_img_pts.push_back(img_pts[k1]);
        sampled_img_pts.push_back(img_pts[k2]);
        sampled_img_pts.push_back(img_pts[k3]);
        sampled_img_pts.push_back(img_pts[k4]);
        
        sampled_wld_pts.push_back(candidate_wld_pts[k1][0]);
        sampled_wld_pts.push_back(candidate_wld_pts[k2][0]);
        sampled_wld_pts.push_back(candidate_wld_pts[k3][0]);
        sampled_wld_pts.push_back(candidate_wld_pts[k4][0]);
        
        Mat rvec;
        Mat tvec;
        bool is_solved = cv::solvePnP(Mat(sampled_wld_pts), Mat(sampled_img_pts), camera_matrix, dist_coeff, rvec, tvec, false, CV_EPNP);
        if (is_solved) {
            rt_candidate.push_back(std::make_pair(rvec, tvec));
        }
        
        if (rt_candidate.size() > K) {
            printf("initialization repeat %d times\n", i);
            break;
        }
    }
    printf("init camera parameter number is %lu\n", rt_candidate.size());
    
    vector<CameraPoseHypothese> losses;
    for (int i = 0; i<rt_candidate.size(); i++) {
        CameraPoseHypothese hyp(0.0);
        hyp.rvec_ = rt_candidate[i].first;
        hyp.tvec_ = rt_candidate[i].second;
        losses.push_back(hyp);
    }
    
    const double threshold = param.reproj_threshold;
    const double line_wt   = param.line_weight;
  
    // data format transformation
    vector<Eigen::Vector2d> begin_points;
    vector<Eigen::Vector2d> end_points;    
    cv2eigen(line_start_points, begin_points);
    cv2eigen(line_end_points, end_points);
             
    
    Eigen::Matrix3d eigen_camera_matrix;
    cv2eigen(camera_matrix, eigen_camera_matrix);
    
    while (losses.size() > 1) {
        // sample random set
        vector<cv::Point2d> sampled_img_pts;
        vector< vector<cv::Point3d> > sampled_wld_pts;  // one camera point may have multiple world points correspondences
        vector<int> sampled_indices;
        for (int i =0; i<B; i++) {
            int index = rand()%N;
            sampled_img_pts.push_back(img_pts[index]);
            sampled_wld_pts.push_back(candidate_wld_pts[index]);
            sampled_indices.push_back(index);
        }
        
        // count outliers
        for (int i = 0; i<losses.size(); i++) {
            // evaluate the accuracy by check re-projection error
            vector<vector<cv::Point2d> > all_projected_pts;
            Mat rvec = losses[i].rvec_;
            Mat tvec = losses[i].tvec_;
            // project all world points to image using estimated rotation and translation vector
            for (int j = 0; j<sampled_wld_pts.size(); j++) {
                vector<cv::Point2d> projected_pts;
                cv::projectPoints(sampled_wld_pts[j], rvec, tvec, camera_matrix, dist_coeff, projected_pts);
                all_projected_pts.push_back(projected_pts);
            }
            assert(all_projected_pts.size() == sampled_img_pts.size());
            
            // check reprojection error
            for (int j = 0; j<sampled_img_pts.size(); j++) {
                double min_dis = threshold * 2;
                int min_index = -1;
                cv::Point2d img_pt = sampled_img_pts[j];
                for (int k = 0; k<all_projected_pts[j].size(); k++) {
                    cv::Point2d dif = img_pt - all_projected_pts[j][k];
                    double dis = cv::norm(dif);
                    if (dis < min_dis) {
                        min_dis = dis;
                        min_index = k;
                    }
                } // end of k
                
                if (min_dis > threshold) {
                    losses[i].loss_ += 1.0;
                }
                else {
                    losses[i].inlier_indices_.push_back(sampled_indices[j]);
                    losses[i].inlier_candidate_world_pts_indices_.push_back(min_index);
                }
            } // end of j
            assert(losses[i].inlier_indices_.size() == losses[i].inlier_candidate_world_pts_indices_.size());
            // printf("inlier number is %lu\n", losses[i].inlier_indices_.size());
            
            Eigen::Affine3d proj3x4 = getProjectionMatrix(camera_matrix, rvec, tvec);
            for (int j = 0; j<candidate_wld_lines.size(); j++) {
                Eigen::ParametrizedLine<double, 2> line2d = project3DLine(proj3x4, candidate_wld_lines[j]);
                double dist1 = line2d.distance(begin_points[j]);
                double dist2 = line2d.distance(end_points[j]);
                
                if (dist1 < threshold && dist2 < threshold) {
                    losses[i].line_indices_.push_back(j);
                  //  cout<<"dist1 "<<dist1<<" dist2 "<<dist2<<endl;
                }
                else {
                    losses[i].loss_ += line_wt;
                }
            }
            
        }
        
        std::sort(losses.begin(), losses.end());
        losses.resize(losses.size()/2);
        
        // refine by inliers
        for (int i = 0; i < losses.size(); i++) {
            // number of inliers is larger than minimum configure
            if (losses[i].inlier_indices_.size() > 4) {
                vector<cv::Point2d> inlier_img_pts;
                vector<cv::Point3d> inlier_wld_pts;
                for (int j = 0; j < losses[i].inlier_indices_.size(); j++) {
                    int index = losses[i].inlier_indices_[j];
                    int wld_index = losses[i].inlier_candidate_world_pts_indices_[j];
                    inlier_img_pts.push_back(img_pts[index]);
                    inlier_wld_pts.push_back(candidate_wld_pts[index][wld_index]);
                }
                // only use points
                if (losses[i].line_indices_.size() == 0) {
                    Mat rvec = losses[i].rvec_;
                    Mat tvec = losses[i].tvec_;
                    bool is_solved = cv::solvePnP(Mat(inlier_wld_pts), Mat(inlier_img_pts), camera_matrix, dist_coeff, rvec, tvec, true, CV_EPNP);
                    if (is_solved) {
                        losses[i].rvec_ = rvec;
                        losses[i].tvec_ = tvec;
                        losses[i].inlier_indices_.clear();
                        losses[i].inlier_candidate_world_pts_indices_.clear();
                        losses[i].line_indices_.clear();
                    }
                }
                else {
                    vector<Vector2d> cur_img_pts;
                    vector<Vector3d> cur_world_pts;
                    cv2eigen(inlier_img_pts, cur_img_pts);
                    cv2eigen(inlier_wld_pts, cur_world_pts);
                    
                    vector<vector<Vector2d > > cur_image_line_pts(losses[i].line_indices_.size());
                    vector<Eigen::ParametrizedLine<double, 3> > cur_world_lines;
                    for (int j = 0; j<losses[i].line_indices_.size(); j++) {
                        int index = losses[i].line_indices_[j];
                        cur_image_line_pts[j].push_back(begin_points[index]);
                        cur_image_line_pts[j].push_back(end_points[index]);
                        cur_world_lines.push_back(candidate_wld_lines[index]);
                    }
                    
                    Eigen::Affine3d init_pose = getAffineMatrix(losses[i].rvec_, losses[i].tvec_);
                    Eigen::Affine3d refined_pose;
                    
                    bool is_optimized = VpglAlgo::estimateCameraPose(cur_img_pts, cur_world_pts,
                                                                     cur_image_line_pts, cur_world_lines,
                                                                     eigen_camera_matrix,
                                                                     init_pose, refined_pose);
                    if (is_optimized) {
                        cv::Mat rvec;
                        cv::Mat tvec;
                        affineToRotationTranslation(refined_pose, rvec, tvec);
                        losses[i].rvec_ = rvec;
                        losses[i].tvec_ = tvec;
                        losses[i].inlier_indices_.clear();
                        losses[i].inlier_candidate_world_pts_indices_.clear();
                        losses[i].line_indices_.clear();
                      //  printf("index is %d\n\n", i);
                    }
                }
            }
        }        
    }
    
    assert(losses.size() == 1);
    
    // change to camera to world transformation
    Mat rot;
    cv::Rodrigues(losses.front().rvec_, rot);
    Mat tvec = losses.front().tvec_;
    assert(tvec.rows == 3);
    assert(tvec.type() == CV_64FC1);
    assert(rot.type() == CV_64FC1);
    assert(rot.rows == 3 && rot.cols == 3);
    
    camera_pose = cv::Mat::eye(4, 4, CV_64F);
    rot.copyTo(camera_pose(cv::Rect(0, 0, 3, 3)));
    
    camera_pose.at<double>(0, 3) = tvec.at<double>(0, 0);
    camera_pose.at<double>(1, 3) = tvec.at<double>(1, 0);
    camera_pose.at<double>(2, 3) = tvec.at<double>(2, 0);
    
    // camere to world coordinate
    camera_pose = camera_pose.inv();
    
    if (isnan(camera_pose.at<double>(0, 0))) {
        return false;
    }
    
    return true;
}