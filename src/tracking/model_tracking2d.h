//
//  cvx_model_tracking2d.h
//  CalibMeMatching
//
//  Created by jimmy on 2018-02-05.
//  Copyright (c) 2018 Nowhere Planet. All rights reserved.
//

#ifndef __CalibMeMatching__cvx_model_tracking2d__
#define __CalibMeMatching__cvx_model_tracking2d__

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <vector>
#include <Eigen/Dense>
#include "pgl_perspective_camera.h"

using std::vector;
using cv::Mat;
using Eigen::Vector2d;


namespace cvx {
    // assume the initial camera is close to the correct camera pose
    // estimate (refine) camera pose by traking an edge image
    // init_camera: initial camera pose
    // model_lines: points in the 2D model, unit meter
    // edge_image: CV_8UC1, edge map
    // refined_camera: refined camera
    // return: true, camera is refined; false, the refined camera is the input camera
    // use RANSAC for line fitting but not for camera parameter estimation
    bool trackEdgeImage(const cvx_pgl::perspective_camera& init_camera,
                        const vector<std::pair<Vector2d, Vector2d> >& model_lines,
                        const cv::Mat& edge_image,
                        cvx_pgl::perspective_camera& refined_camera);
    
    // implement the edge search method in paper "Automatic rectification of long image sequences"
    // ACCV 2004
    // init_camera: initial camera pose
    // model_points: points in the 2D model, unit meter
    // image: CV_8UC1, input image
    // search_distance: distance in the normal direction, unit meter, 1.0. model coordinate
    // refined_camera: refined camera
    // return: true, camera is refined; false, the refined camera is the input camera
    // use RANSAC inside
    bool edgeSearch(const cvx_pgl::perspective_camera& init_camera,
                    const vector<Vector2d>& model_points,
                    const vector<Vector2d>& model_point_normal_direction,
                    const cv::Mat& image,
                    const double search_distance,
                    cvx_pgl::perspective_camera& refined_camera);
    
}

#endif /* defined(__CalibMeMatching__cvx_model_tracking2d__) */
