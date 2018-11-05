//
//  cvx_pgl_perspective_camera.h
//  CalibMeMatching
//
//  Created by jimmy on 2018-02-05.
//  Copyright (c) 2018 Nowhere Planet. All rights reserved.
//

#ifndef __CalibMeMatching__cvx_pgl_perspective_camera__
#define __CalibMeMatching__cvx_pgl_perspective_camera__

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include "pgl_perspective_camera.h"

using Eigen::Vector2d;
using std::vector;

namespace cvx_pgl {
    // @brief refine camera using point-to-point and point-on-line correspondences
    // model_pts: points from 2D model
    // im_pts: points from destination image
    // model_lines: line (edge) from 2D model
    // im_line_pts: sampled points on line from destination image
    // init_camera: initial camera
    // refined_camera: refined camera
    // fix principal point
    // return: mean reporjection error
    double estimateCamera(const vector<Vector2d> & model_pts,
                          const vector<Vector2d> & im_pts,
                          const vector<std::pair<Eigen::Vector2d, Eigen::Vector2d> > & model_lines,
                          const vector<Vector2d> & im_line_pts,
                          const perspective_camera& init_camera,
                          perspective_camera& refined_camera);
    
    // RANSAC: using point on line estimation
    bool estimateCameraRANSAC(const vector<std::pair<Eigen::Vector2d, Eigen::Vector2d> > & model_lines,
                              const vector<Vector2d> & im_line_pts,
                              const perspective_camera& init_camera,
                              perspective_camera& refined_camera);
    
}

#endif /* defined(__CalibMeMatching__cvx_pgl_perspective_camera__) */
