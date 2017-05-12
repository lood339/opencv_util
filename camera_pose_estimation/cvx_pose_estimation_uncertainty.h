//
//  cvx_pose_estimation_uncertainty.h
//  PointLineReloc
//
//  Created by jimmy on 2017-05-01.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __PointLineReloc__cvx_pose_estimation_uncertainty__
#define __PointLineReloc__cvx_pose_estimation_uncertainty__

#include <stdio.h>
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <string>

using std::vector;
using std::unordered_map;
using std::string;

class CvxPoseEstimationUncertainty
{
public:
    struct PreemptiveRANSACUncParameter
    {
        double dist_threshold_;    // distance threshod, unit meter
        int sample_num_;           // number of samples of camera points
        double line_point_wt_;     // line point weight
        int min_line_support_num_;    // minimum number of points to support a line
        int line_sample_num_;
        double min_variance_;       // prevent singular covariance matrix, 0.005 meter
        int point_line_joint_opt_num_;   // hypothesis that use line and point, small number speed up
        
    public:
        PreemptiveRANSACUncParameter()
        {
            dist_threshold_ = 0.1;  // meter
            sample_num_ = 500;
            line_point_wt_ = 0.5;
            min_line_support_num_ = 6;
            line_sample_num_ = 10;
            min_variance_ = 0.005;
            point_line_joint_opt_num_ = 4;
        }
        
        bool readFromFile(const char *file)
        {
            FILE *pf = fopen(file, "r");
            assert(pf);
            const int param_num = 7;
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
            
            dist_threshold_ = imap[string("dist_threshold")];
            sample_num_ = imap[string("sample_num")];
            line_point_wt_ = imap[string("line_point_wt")];
            min_line_support_num_ = imap[string("min_line_support_num")];
            line_sample_num_ = (int)imap[string("line_sample_num")];
            min_variance_ = imap[string("min_variance")];
            point_line_joint_opt_num_ = (int)imap[string("point_line_joint_opt_num")];
            
            fclose(pf);
            return true;
        }

    };
    
public:
    // world_pts_covariance: covariance matrix, make sure they are invertable
    static bool preemptiveRANSAC3DOneToMany(const vector<Eigen::Vector3d>& camera_pts,
                                            const vector<vector<Eigen::Vector3d> >& world_pts,
                                            const vector<vector<Eigen::Matrix3d> >& world_pts_covariance,
                                            const PreemptiveRANSACUncParameter & param,
                                            Eigen::Affine3d& camera_pose);
    
    // camera pose estimation using both points and lines
    static bool preemptiveRANSACPointAndLine(const vector<Eigen::Vector3d>& camera_pts,
                                             const vector<vector<Eigen::Vector3d> >& world_pts,
                                             const vector<vector<Eigen::Matrix3d> >& world_pts_covariance,
                                             
                                             const vector<vector< Eigen::Vector3d> > & world_line_pts_group,
                                             const vector<vector< Eigen::Matrix3d> > & world_line_pts_covariance,
                                             const vector<std::pair<Eigen::Vector3d, Eigen::Vector3d> >& camera_lines,
                                             const PreemptiveRANSACUncParameter & param,
                                             Eigen::Affine3d& camera_pose);
    
private:
    // sample camera poses from 4 points corresondences
    static
    vector<Eigen::Affine3d> sampleCameraPose(const vector<Eigen::Vector3d>& camera_pts,
                                             const vector<vector<Eigen::Vector3d> >& world_pts,
                                             const int num_iteration, const int num_poses);
};

#endif /* defined(__PointLineReloc__cvx_pose_estimation_uncertainty__) */
