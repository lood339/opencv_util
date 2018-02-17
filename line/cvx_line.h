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
    struct ZhangLine3D {
        // notation from Zhang's paper "Determining motion from 3D line segment matches: A comparative study"
        Eigen::Vector3d u;  // unit direction, segment has direction
        Eigen::Vector3d d;  // norm of d is the distance from the origin to line, the direction of d is parallel
                            // the normal of plane containing the line and the origin
        ZhangLine3D(){}
        
        //m1, m2: two points on the line, order maters
        ZhangLine3D(const Eigen::Vector3d& m1, const Eigen::Vector3d& m2)
        {
            Eigen::Vector3d l = m2 - m1;
            Eigen::Vector3d m = (m1 + m2)/2.0;
            
            CvxLine::ZhangLine3D line;
            if(l.norm() != 0) {
                u = l/l.norm();
                d = l.cross(m)/l.norm();
            }
        }

    };
    // shift end points of (p3, p4) along the line p3 --> p4, so that p1 close to p3 and p2 close to p4
    // assume line direction p1 --> p2, p3 --> p4
    static void shiftEndPoints(const Eigen::Vector3d & p1, const Eigen::Vector3d & p2,
                               Eigen::Vector3d & p3, Eigen::Vector3d & p4);
    
    // project p1...p4 to lines
    // line1: p1--> p2
    // line2: p3--> p4
    // projected_points: the projection of p1, p2, p3, p4 to line1 line2, respectively
    static void projectEndPoints(const Eigen::Vector3d & p1, const Eigen::Vector3d & p2,
                                 const Eigen::Vector3d & p3, const Eigen::Vector3d & p4,
                                 vector<Eigen::Vector3d> & projected_points);
    
    // motion [R:T] from two sets of line segment correspondences
    static bool motionFromLines(const vector<ZhangLine3D>& source_lines,
                                const vector<ZhangLine3D>& dst_lines,
                                Eigen::Matrix3d& rotation,
                                Eigen::Vector3d& translation);
};

// tempalte version, VectorT can be Eigen::Vector3d or Eigen::Vector2d
// shift end points of (p3, p4) along the line p3 --> p4, so that p1 close to p3 and p2 close to p4
// assume line direction p1 --> p2, p3 --> p4
template <class VectorT>
void shiftLineEndPoints(const VectorT & p1, const VectorT & p2, VectorT & p3, VectorT & p4);

// pixel location along line p1 --> p2
bool getLinePixels(const Eigen::Vector2d & p1, const Eigen::Vector2d & p2, vector<Eigen::Vector2d > & line_points);

// assume all point are inliers
bool fitLine3D(const vector<Eigen::Vector3d > & line_points, Eigen::ParametrizedLine<double, 3> & line);



bool fitLine3DRansac(const vector<Eigen::Vector3d > & line_points, Eigen::ParametrizedLine<double, 3> & line);

bool fitLine3DRansac(const vector<Eigen::Vector3d > & line_points,
                     Eigen::ParametrizedLine<double, 3> & line,
                     vector<unsigned>& inlier_index);

bool fitLine2DRansac(const vector<Eigen::Vector2d> & points,
                     Eigen::ParametrizedLine<double, 2> & line,
                     vector<int>& inlier_index,
                     const double inlier_distance_threshold = 1.0); 

bool fitLine3DRansac(const vector<Eigen::Vector3d > & line_points, Eigen::ParametrizedLine<double, 3> & line,
                     std::pair<Eigen::Vector3d, Eigen::Vector3d>& line_end_point);


// Accurate and linear time pose estimation from points and lines eccv 2016,
// equation (6)
// not very accurate, re-projection error can be upto 3-4 pixels
void backprojectImagePointTo3Dline(const vector<Eigen::Vector2d>& img_pts,
                                   const vector<Eigen::ParametrizedLine<double, 3> >& camera_lines,
                                   const Eigen::Matrix3d& camera_matrix,
                                   vector<Eigen::Vector3d>& points_on_line);

// Robust RGBD ordometry using point and line feature. ICCV 2015
// Lu's line uncertainty, equation (6)
// labmda: sigma_p^{-1}
Eigen::Matrix3d JacobQwithA(const Eigen::Vector3d& A, const Eigen::Vector3d& B,
                            const Eigen::Vector3d& P, const Eigen::Matrix3d& lambda);

Eigen::Matrix3d JacobQwithB(const Eigen::Vector3d& A, const Eigen::Vector3d& B,
                            const Eigen::Vector3d& P, const Eigen::Matrix3d& lambda);

// ordered_point_precision: invert of covariance matrix
// represent a line with two points [A; B], cov_line is a 6 x 6 matrix
bool fitLuLine3D(const vector<Eigen::Vector3d> & ordered_line_points,
                 const vector<Eigen::Matrix3d> & ordered_point_precision,
                 Eigen::Vector3d& A, Eigen::Vector3d& B, Eigen::MatrixXd & cov_line);

// input covariance instead of precision matrix
bool fitLuLine3D(const vector<Eigen::Vector3d> & ordered_line_points,
                 const vector<Eigen::Matrix3d> & ordered_point_covariance,
                 Eigen::Vector3d& A, Eigen::Vector3d& B, Eigen::MatrixXd & cov_line,
                 bool check_invert);

// compute distance variance from a point to a 3D line
// D2 = distance ^2, squared norm, distance: from a point C to a line represented by [A; B]
Eigen::Vector3d JacobD2withA(const Eigen::Vector3d& A, const Eigen::Vector3d& B,
                             const Eigen::Vector3d& C);

Eigen::Vector3d JacobD2withB(const Eigen::Vector3d& A, const Eigen::Vector3d& B,
                             const Eigen::Vector3d& C);

// line pass [A B], error forward propagation
bool pointToLineDistanceUncertainty(const Eigen::Vector3d& A, const Eigen::Vector3d& B,
                                    const Eigen::MatrixXd& cov_line,
                                    const Eigen::Vector3d& P,
                                    double& distance,
                                    double& sigma);





#endif /* defined(__PointLineReloc__eigen_line_matching__) */
