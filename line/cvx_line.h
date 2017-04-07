//
//  eigen_line_matching.h
//  PointLineReloc
//
//  Created by jimmy on 2017-03-29.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __PointLineReloc__eigen_line_matching__
#define __PointLineReloc__eigen_line_matching__

// 3D line matching from "accurate and linear time pose estimation from points and lines", eccv 2016

#include <stdio.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>

using std::vector;


class CvxLine
{
public:
    // shift end points of (p3, p4) along the line p3 --> p4, so that p1 close to p3 and p2 close to p4
    // assume line direction p1 --> p2, p3 --> p4
    static void shiftEndPoints(const Eigen::Vector3d & p1, const Eigen::Vector3d & p2,
                               Eigen::Vector3d & p3, Eigen::Vector3d & p4);
    
    // project p1...p4 to lines
    // line1: p1--> p2
    // line2: p3--> p3
    // projected_points: the projection of p1, p2, p3, p4 to line1 line2, respectively
    static void projectEndPoints(const Eigen::Vector3d & p1, const Eigen::Vector3d & p2,
                                 const Eigen::Vector3d & p3, const Eigen::Vector3d & p4,
                                 vector<Eigen::Vector3d> & projected_points);
    
};

bool getLinePixels(const Eigen::Vector2d & p1, const Eigen::Vector2d & p2, vector<Eigen::Vector2d > & line_points);

bool fitLine3D(const vector<Eigen::Vector3d > & line_points, Eigen::ParametrizedLine<double, 3> & line);

bool fitLine3DRansac(const vector<Eigen::Vector3d > & line_points, Eigen::ParametrizedLine<double, 3> & line);

#endif /* defined(__PointLineReloc__eigen_line_matching__) */
