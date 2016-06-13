//
//  cvxPoseEstimation.hpp
//  LoopClosure
//
//  Created by jimmy on 2016-03-31.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef cvxPoseEstimation_cpp
#define cvxPoseEstimation_cpp

#include <stdio.h>
#include "cvxImage_310.hpp"
#include <vector>
#include <vil/vil_image_view.h>

using std::vector;

struct PreemptiveRANSACParameter
{
    double reproj_threshold; // re-projection error threshold, unit pixel
    
public:
    PreemptiveRANSACParameter()
    {
        reproj_threshold = 15.0;
    }
};

struct PreemptiveRANSAC3DParameter
{
    double dis_threshold_; // distance threshod, unit meter
public:
    PreemptiveRANSAC3DParameter()
    {
        dis_threshold_ = 0.1;
    }
    
};

class CvxPoseEstimation
{
public:
    // SCRF_testing_result: predictions from random forests
    // pts_camera_view: correspondent camera view 3D coordinate
    // camera_pose: camera to world, 4 * 4 matrix
    // outlier_threshold: unit of pixel
  
    // 3D - 2D
    static bool estimateCameraPose(const cv::Mat & camera_matrix,
                                   const cv::Mat & dist_coeff,
                                   const vector<cv::Point2d> & im_pts,
                                   const vector<cv::Point3d> & wld_pts,
                                   cv::Mat & camera_pose,
                                   const double outlier_threshold = 8.0);
    
    // pose estimation from a nearby location orientation image
    // using sift matching
    static bool estimateCameraPoseFromImageMatching(
                                                    const cv::Mat & camera_matrix,
                                                    const cv::Mat & dist_coeff,
                                                    const vil_image_view<vxl_byte> & query_rgb_image,
                                                    const vil_image_view<vxl_byte> & database_rgb_image,
                                                    const cv::Mat & database_depth_image,
                                                    const cv::Mat & database_pose,
                                                    cv::Mat & estimated_pose,
                                                    const int min_matching_num = 50);
    
    // wld_pts: estimated points, has outliers
    static bool preemptiveRANSAC(const vector<cv::Point2d> & img_pts,
                                 const vector<cv::Point3d> & wld_pts,
                                 const cv::Mat & camera_matrix,
                                 const cv::Mat & dist_coeff,
                                 const PreemptiveRANSACParameter & param,
                                 cv::Mat & camera_pose);
    
    // wld_pts: estimated points, had outliers
    static bool preemptiveRANSAC3D(const vector<cv::Point3d> & camera_pts,
                                   const vector<cv::Point3d> & wld_pts,
                                   const PreemptiveRANSAC3DParameter & param,
                                   cv::Mat & camera_pose);
    
    // angle_distance: degree
    static void poseDistance(const cv::Mat & src_pose,
                      const cv::Mat & dst_pose,
                      double & angle_distance,
                      double & euclidean_disance);
    
    // 3x3 rotation matrix to eular angle
    static Mat rotationToEularAngle(const cv::Mat & rot);
    
    // return CV_64FC1 4*1
    static Mat rotationToQuaternion(const cv::Mat & rot);
    
    
    
    static inline float SIGN(float x) {return (x >= 0.0f) ? +1.0f : -1.0f;}
    static inline float NORM(float a, float b, float c, float d) {return sqrt(a * a + b * b + c * c + d * d);}
};

#endif /* cvxPoseEstimation_cpp */
