//
//  cvx_line_pose_estimation.cpp
//  PointLineReloc
//
//  Created by jimmy on 2017-04-23.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "cvx_line_pose_estimation.h"
#include "cvx_line.h"
#include "vnl_algo.h"


bool CvxLinePoseEstimation::preemptiveRANSAC3DOneToMany(const vector<vector<Eigen::Vector3d> >& world_line_pts,
                                                        const vector<vector<Eigen::Vector3d> >& camera_line_pts,
                                                        const vector<std::pair<Eigen::Vector3d, Eigen::Vector3d> > & world_line_segments,
                                                        const vector<std::pair<Eigen::Vector3d, Eigen::Vector3d> > & camera_line_segments,
                                                        const PreemptiveRANSACLine3DParameter & param,
                                                        Eigen::Affine3d& output_camera_pose)
{
    assert(world_line_pts.size() == world_line_segments.size());
    assert(world_line_pts.size() == camera_line_segments.size());
    assert(world_line_pts.size() == camera_line_pts.size());    
    
    const int num_iteration = 2048;
    const int K = 512;
    const int N = (int)world_line_pts.size();
    
    if (N <= 4) {
        return false;
    }
    
    // initial camera pose
    vector<Eigen::Affine3d > affine_candidate;
    for (int i = 0; i<num_iteration; i++) {
        // (1) random pick three lines
        int random_index[3] = {0};
        do{
            random_index[0] = rand()%N;
            random_index[1] = rand()%N;
            random_index[2] = rand()%N;
        }while (random_index[0] == random_index[1]
                || random_index[0] == random_index[2]
                || random_index[1] == random_index[2]);
        
        vector<CvxLine::ZhangLine3D> source_lines;
        vector<CvxLine::ZhangLine3D> dest_lines;
        
        for (int j = 0; j < 3; j++) {
            int index = random_index[j];
            source_lines.push_back(CvxLine::ZhangLine3D(world_line_segments[index].first, world_line_segments[index].second));
            dest_lines.push_back(CvxLine::ZhangLine3D(camera_line_segments[index].first, camera_line_segments[index].second));
        }
        
        
        // (2) initial camera pose estimation using Zhang's method
        Eigen::Matrix3d rotation;
        Eigen::Vector3d translation;
        Eigen::Affine3d init_pose;
        CvxLine::motionFromLines(source_lines, dest_lines, rotation, translation);
        init_pose.linear() = rotation;
        init_pose.translation() = translation;
        
        affine_candidate.push_back(init_pose);
        if (affine_candidate.size() > K) {
            break;
        }
    }
    
    
    // prepared camera line equation
    vector<Eigen::ParametrizedLine<double, 3> > camera_lines;
    for (int i = 0; i<camera_line_segments.size(); i++) {
        Eigen::Vector3d p1 = camera_line_segments[i].first;
        Eigen::Vector3d p2 = camera_line_segments[i].second;
        
        camera_lines.push_back(Eigen::ParametrizedLine<double, 3>::Through(p1, p2));
    }
    
    // check the inlier number of each candidate affine transformation
    const double point_dist_threshold = param.point_dist_threshold;
    const double line_dist_threshold = param.point_to_line_dist_threshold;
    const int inlier_line_num_threshold = 3;
    int max_inlier = 0;
    int best_index = -1;
    // loop each candidate
    for (int c = 0; c<affine_candidate.size(); c++) {
        Eigen::Affine3d cur_affine = affine_candidate[c];
        
        int cur_point_inlier_num = 0;  // inlier number of world coordinate points
        int cur_line_inlier_num = 0;
        vector<vector<Eigen::Vector3d> > inlier_wld_pts_group;
        vector<Eigen::Vector3d> inlier_world_point;
        vector<Eigen::Vector3d> inlier_camera_point;
        // loop each line
        for (int j = 0; j<camera_lines.size(); j++) {
            vector<Eigen::Vector3d> inlier_wld_pts;
            for (int k = 0; k < world_line_pts[j].size(); k++) {
                Eigen::Vector3d p = world_line_pts[j][k];
                Eigen::Vector3d q = cur_affine * p;  // project from world coordinate to camera coordinate
                
                double dist1 = camera_lines[j].distance(q);
                double dist2 = (camera_line_pts[j][k] - q).norm();
                
                if (dist1 < line_dist_threshold) {
                    inlier_wld_pts.push_back(p);
                }
                if (dist2 < point_dist_threshold) {
                    inlier_world_point.push_back(p);
                    inlier_camera_point.push_back(camera_line_pts[j][k]);
                }
                if (dist1 < line_dist_threshold || dist2 < point_dist_threshold ) {
                    cur_point_inlier_num++;
                }
            }
            if (inlier_wld_pts.size() > 3) {
                cur_line_inlier_num++;
            }
            inlier_wld_pts_group.push_back(inlier_wld_pts);
        }
        
        
        if (cur_line_inlier_num > inlier_line_num_threshold &&
            cur_point_inlier_num > max_inlier) {
            max_inlier = cur_point_inlier_num;
            best_index = c;
            
            assert(inlier_wld_pts_group.size() == camera_lines.size());
            
            // refine camera pose
            Eigen::Affine3d refined_pose;
            bool is_refined = VnlAlgo::estimateCameraPoseUsingPointLine(inlier_world_point, inlier_camera_point, inlier_wld_pts_group, camera_lines, cur_affine, refined_pose);
            if (is_refined) {
                affine_candidate[c] = refined_pose;
            }
            printf("inlier point, line number: %d %d \n", cur_point_inlier_num, cur_line_inlier_num);
        }
    }
    
    // no camera is refined
    if (best_index == -1) {
        return false;
    }
    
    output_camera_pose = affine_candidate[best_index];
    return true;
}