//
//  cvx_rgbd.h
//  PointLineReloc
//
//  Created by jimmy on 2017-03-30.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __PointLineReloc__cvx_rgbd__
#define __PointLineReloc__cvx_rgbd__

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <string>
#include <vector>
#include <Eigen/Dense>

using Eigen::Vector3d;
using Eigen::Vector2d;
using std::vector;

class CvxRGBD
{
public:
    struct RGBDLineSegment
    {    
        Vector2d img_p1;
        Vector2d img_p2;
        
        Vector3d cam_p1;
        Vector3d cam_p2;
        
        Vector3d wld_p1;
        Vector3d wld_p2;
    };
    struct RGBDLineParameter
    {
        double min_length;        // 2D image space, pixel
        double min_camera_points; // 3D camera coordiante points, number
        double inlier_point_threshold; // 3D point to estiamted 3D line, meter
        double inlier_ratio;           // 3D point on 3D line, percentage
        double sample_densitiy;        // sample point on the line, (0, 1.0)
        double line_area_width;        // line area with, pixel, calculate line direction, bright in left side
        double brightness_contradict_threshold; // brightness contradict of left and right side of linesegment
        
        RGBDLineParameter()
        {
            min_length = 40.0;
            min_camera_points = 15.0;
            inlier_point_threshold = 0.02;
            inlier_ratio = 0.5;
            sample_densitiy = 0.5;
            line_area_width = 15;
            brightness_contradict_threshold = 0.02;
        }
    };
    
public:
    // camera_depth_img: CV_64FC1
    // camera_to_world_pose: 4x4 CV_64FC1
    // calibration_matrix: 3x3 CV_64FC1
    // depth_factor: e.g. 1000.0 for MS 7 scenes
    // camera_xyz: output camera coordinate location, CV_64FC3
    // world_coordinate: CV_64FC3 , x y z in world coordinate
    // mask: output CV_8UC1 0 -- > invalid, 1 --> valid
    static bool cameraDepthToWorldCoordinate(const cv::Mat & camera_depth_img,
                                             const cv::Mat & camera_to_world_pose,
                                             const cv::Mat & calibration_matrix,
                                             const double depth_factor,
                                             const double min_depth,
                                             const double max_depth,
                                             cv::Mat & camera_coordinate,
                                             cv::Mat & world_coordinate,
                                             cv::Mat & mask);
    
    // depth image to camera coordinate
    static bool cameraDepthToCameraCoordinate(const cv::Mat & camera_depth_img,
                                             const cv::Mat & calibration_matrix,
                                             const double depth_factor,
                                             const double min_depth,
                                             const double max_depth,
                                             cv::Mat & camera_coordinate,
                                             cv::Mat & mask);
    
    // color_img: CV_8UC3 in default BGR format
    // detect 3D lines in camera coordinate
    // depth_img: CV_64FC1, in meter
    // mask: CV_8UC1, 0 or 1
    // calibration_matrix: camera intrinsic parameter, check line segment end points in image space
    // camera_coordinate: CV_64FC3
    // return: line segment in image coordinate and camera coordinate
    // todo: RANSAC 3D line estimation, wider lines
    static bool detect3DLines(const cv::Mat & color_img,
                              const cv::Mat & depth_img,
                              const cv::Mat & mask,
                              const cv::Mat & calibration_matrix,
                              const cv::Mat & camera_coordinate,
                              const RGBDLineParameter & line_param,
                              vector<RGBDLineSegment> & lines);
    
    
    // depth_img: input and output
    static bool depthInpaint(cv::Mat & depth_img,
                             const unsigned char no_depth_mask);

    
};

#endif /* defined(__PointLineReloc__cvx_rgbd__) */
