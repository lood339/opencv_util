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

struct CameraPoseHypothese
{
    double loss_;
    Mat rvec_;       // rotation     for 3D --> 2D projection
    Mat tvec_;       // translation  for 3D --> 2D projection
    Mat affine_;     //              for 3D --> 3D camera to world transformation
    vector<unsigned int> inlier_indices_;         // camera coordinate index
    vector<unsigned int> inlier_candidate_world_pts_indices_; // candidate world point index
    
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
        std::copy(other.camera_pts_.begin(), other.camera_pts_.end(), camera_pts_.begin());
        std::copy(other.wld_pts_.begin(), other.wld_pts_.end(), wld_pts_.begin());
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
        std::copy(other.camera_pts_.begin(), other.camera_pts_.end(), camera_pts_.begin());
        std::copy(other.wld_pts_.begin(), other.wld_pts_.end(), wld_pts_.begin());
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
    
    const double threshold = param.dis_threshold_;
    const double line_threshold = param.line_distance_threshold;
    const double line_loss_weight = param.line_loss_weight;
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
                }
                else {
                    losses[i].loss_ += line_loss_weight;  // todo, the weight of lines
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
                }
                else {
                     CvxCalib3D::KabschTransform(inlier_camera_pts, inlier_wld_pts, affine);                    
                }                
               
                losses[i].affine_ = affine;
                losses[i].inlier_indices_.clear();
                losses[i].inlier_candidate_world_pts_indices_.clear();
                losses[i].camera_pts_.clear();
                losses[i].wld_pts_.clear();
            }
        }
    }
    assert(losses.size() == 1);
    
    camera_pose = cv::Mat::eye(4, 4, CV_64F);
    losses[0].affine_.copyTo(camera_pose(cv::Rect(0, 0, 4, 3)));
    return true;
}