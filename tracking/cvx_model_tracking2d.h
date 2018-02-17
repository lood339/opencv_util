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
    // use RANSAC inside
    bool trackEdgeImage(const cvx_pgl::perspective_camera& init_camera,
                        const vector<std::pair<Vector2d, Vector2d> >& model_lines,
                        const cv::Mat& edge_image,
                        cvx_pgl::perspective_camera& refined_camera);
    
}

#endif /* defined(__CalibMeMatching__cvx_model_tracking2d__) */