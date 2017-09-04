//
//  cvx_gl_homography.h
//  CalibMeMatching
//
//  Created by jimmy on 2017-07-30.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __CalibMeMatching__cvx_gl_homography__
#define __CalibMeMatching__cvx_gl_homography__

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>

using Eigen::Vector2d;
using std::vector;

namespace cvx_gl {
    // @brief refine homography using point-to-point and point-on-line correspondences
    // src_pts: points from source image
    // dst_pts: points from destination image
    // src_lines: line (edge) from source image
    // dst_line_pts: sampled points on line from destination image
    // init_homo: 3x3 initial homography
    // refined_homo: refined homography
    bool estimateHomography(const vector<Vector2d> & src_pts,
                            const vector<Vector2d> & dst_pts,
                            const vector<Eigen::ParametrizedLine<double, 2> > & src_lines,
                            const vector<Vector2d> & dst_line_pts,
                            const Eigen::Matrix3d& init_homo,
                            Eigen::Matrix3d& refined_homo);
    
    //@brief refine homography using point-to-point and point-on-line correspondences
    //       from multiple images
    // assume souece image cameras are known
    // model_to_image_homos: model to image homography from all source image cameras
    bool estimateHomography(const vector<vector<Vector2d> > & src_pts,
                            const vector<vector<Vector2d> > & dst_pts,
                            const vector<vector<Eigen::ParametrizedLine<double, 2> > > & src_lines,
                            const vector<vector<Vector2d> > & dst_line_pts,
                            const vector<Eigen::Matrix3d>& model_to_image_homos,
                            const Eigen::Matrix3d& init_homo,
                            Eigen::Matrix3d& refined_homo);
    
    
}

#endif /* defined(__CalibMeMatching__cvx_gl_homography__) */
