//
//  eigen_line_matching.cpp
//  PointLineReloc
//
//  Created by jimmy on 2017-03-29.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "cvx_line.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <iostream>
#include <Eigen/QR>
#include "eigen_geometry_util.h"

using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::ParametrizedLine;
using std::cout;
using std::endl;

void CvxLine::shiftEndPoints(const Eigen::Vector3d & p1, const Eigen::Vector3d & p2,
                                       Eigen::Vector3d & p3, Eigen::Vector3d & p4)
{
    // keep the distance
    double dist = (p3 - p4).norm();
    assert(dist > 0);
    Vector3d v = p4 - p3;
    v.normalize();   // direction
    
    // equation 5 in eccv paper, accurate and linear time pose estimation from points and lines
    double gamma = v.dot(0.5*(p1 + p2) - p3) - dist/2.0;
   
    Vector3d p0 = p3;  // temporal copy
    p3 = p0 + gamma * v;
    p4 = p0 + (gamma + dist) * v;
    
    //printf("gamma, distances are %lf %lf\n", gamma, dist);
}

void CvxLine::projectEndPoints(const Eigen::Vector3d & p1, const Eigen::Vector3d & p2,
                                         const Eigen::Vector3d & p3, const Eigen::Vector3d & p4,
                                         vector<Eigen::Vector3d> & projected_points)
{
    ParametrizedLine<double, 3> line1 = ParametrizedLine<double, 3>::Through(p1, p2);
    ParametrizedLine<double, 3> line2 = ParametrizedLine<double, 3>::Through(p3, p4);
    
    projected_points.resize(4);
    projected_points[0] = line2.projection(p1);
    projected_points[1] = line2.projection(p2);
    projected_points[2] = line1.projection(p3);
    projected_points[3] = line1.projection(p4);    
}


bool CvxLine::motionFromLines(const vector<ZhangLine3D>& source_lines,
                              const vector<ZhangLine3D>& dst_lines,
                              Eigen::Matrix3d & output_rotation, Eigen::Vector3d & output_translation)
{
    assert(source_lines.size() == dst_lines.size());
    assert(source_lines.size() >= 3);
    
    // step 1: estimate rotation
    Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
    for (unsigned int i = 0; i<source_lines.size(); i++) {
        // equation (35)
        Eigen::Matrix4d Ai = Eigen::Matrix4d::Zero();
        Eigen::Vector3d u_dif = source_lines[i].u - dst_lines[i].u;
        Ai(0, 1) = u_dif.x();
        Ai(0, 2) = u_dif.y();
        Ai(0, 3) = u_dif.z();
        Ai(1, 0) = - u_dif.x();
        Ai(2, 0) = - u_dif.y();
        Ai(3, 0) = - u_dif.z();
        
        Eigen::Matrix3d u_src_mat = EigenGeometryUtil::vector2SkewSymmetricMatrix(source_lines[i].u);
        Eigen::Matrix3d u_dst_mat = EigenGeometryUtil::vector2SkewSymmetricMatrix(dst_lines[i].u);
        Eigen::Matrix3d u_src_dst = u_src_mat+ u_dst_mat;
        Ai.block(1, 1, 3, 3) = u_src_dst;  // set sub matrix (right bottom 3 x 3)
        A = A + Ai.transpose() * Ai;
    }
    
    Eigen::JacobiSVD<Eigen::Matrix4d> svd_A(A, Eigen::ComputeFullV);
    Eigen::Vector4d q_v = svd_A.matrixV().col(3);                    // smallest eigen vector
    // w, x, y, z
    Eigen::Quaternion<double> qt(q_v[0], q_v[1], q_v[2], q_v[3]);
    output_rotation = qt.toRotationMatrix();
    
    // step 2: estimate translation, equation (39)
    Eigen::Matrix3d uu = Eigen::Matrix3d::Zero();
    Eigen::Vector3d udr = Eigen::Vector3d::Zero();
    for (unsigned int i = 0; i<dst_lines.size(); i++) {
        Eigen::Matrix3d u_dst_mat = EigenGeometryUtil::vector2SkewSymmetricMatrix(dst_lines[i].u);
        Eigen::Matrix3d ut_dst_mat = u_dst_mat.transpose();
        
        uu += u_dst_mat * ut_dst_mat;   // accumulate all lines
        
        Eigen::Vector3d d_src = source_lines[i].d;
        Eigen::Vector3d d_dst = dst_lines[i].d;
        Eigen::Vector3d d_dif = d_dst - output_rotation * d_src;  // measure in destimation space
        Eigen::Vector3d cur_udr = ut_dst_mat * d_dif;
        
        udr += cur_udr;  // accumulate all lines
    }
    
    output_translation = uu.inverse() * udr;
    return true;
}

template <class VectorT>
void shiftLineEndPoints(const VectorT & p1, const VectorT & p2, VectorT & p3, VectorT & p4)
{
    // keep the distance
    double dist = (p3 - p4).norm();
    assert(dist > 0);
    VectorT v = p4 - p3;
    v.normalize();   // direction
    
    // equation 5 in eccv paper, accurate and linear time pose estimation from points and lines
    double gamma = v.dot(0.5*(p1 + p2) - p3) - dist/2.0;
    
    VectorT p0 = p3;  // temporal copy
    p3 = p0 + gamma * v;
    p4 = p0 + (gamma + dist) * v;
}

// http://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
// Bresenham's line algorithm
bool getLinePixels(const Eigen::Vector2d & p0, const Eigen::Vector2d & p1, vector<Eigen::Vector2d > & line_points)
{
    double x0 = p0.x();
    double y0 = p0.y();
    double x1 = p1.x();
    double y1 = p1.y();
    if (p0.x() > p1.x()) {
        std::swap(x0, x1);
        std::swap(y0, y1);
    }
    
    double deltaX = x1 - x0;
    double deltaY = y1 - y0;
    double error = 0;
    if (fabs(deltaX) < 0.5) {
        // vertical line
        if (y0 > y1) {
            std::swap(y0, y1);
        }
        int x = (x0 + x1)/2.0;
        for (int y = y0; y <= y1; y++) {
            line_points.push_back(Vector2d(x, y));
        }
    }
    else if(fabs(deltaY) < 0.5)
    {
        // horizontal line
        int y = (y0 + y1)/2.0;
        for (int x = x0; x <= x1; x++) {
            line_points.push_back(Vector2d(x, y));
        }
    }
    else
    {
        double deltaErr = fabs(deltaY/deltaX);
        int y = (int)y0;
        int sign = y1 > y0 ? 1:-1;
        for (int x = x0; x <= x1; x++) {
            line_points.push_back(Vector2d(x, y));
            error += deltaErr;
            while (error >= 0.5) {
                line_points.push_back(Vector2d(x, y));  // may have duplicated (x, y)
                y += sign;
                error -= 1.0;
            }
        }
    }
    return true;
}

bool fitLine3D(const vector<Eigen::Vector3d> & line_points, Eigen::ParametrizedLine<double, 3> & output_line)
{
    vector<cv::Point3d> points(line_points.size());
    for (int i = 0; i<line_points.size(); ++i) {
        points[i] = cv::Point3d(line_points[i].x(), line_points[i].y(), line_points[i].z());
    }
    
    cv::Vec6d line3d;
    cv::fitLine(points, line3d, CV_DIST_L2, 0, 0.01, 0.01);
    
    Eigen::Vector3d org_point(line3d[3], line3d[4], line3d[5]);
    Eigen::Vector3d direction(line3d[0], line3d[1], line3d[2]);
    
    output_line = Eigen::ParametrizedLine<double, 3>(org_point, direction);
    
    return true;
}

bool fitLine3DRansac(const vector<Eigen::Vector3d > & line_points, Eigen::ParametrizedLine<double, 3> & output_line)
{
    if(line_points.size() <= 3) {
        return false;
    }
    const int num_iteration = 100;
    const int N = (int)line_points.size();
    const double inlier_distance_threshold = 0.05;
    double min_inlier_ratio = 0.5;
    int best_inlier_num = 0;
    
    Eigen::ParametrizedLine<double, 3> best_line;
    for (int i = 0; i<num_iteration; i++) {
        // randomly pick 2 points and fit a line
        int k1 = 0;
        int k2 = 0;
        
        do{
            k1 = rand()%N;
            k2 = rand()%N;
        }while (k1 == k2);
        
        Eigen::ParametrizedLine<double, 3> min_config_line = Eigen::ParametrizedLine<double, 3>::Through(line_points[k1], line_points[k2]);
        int inlier_num = 0;
        vector<int> inlier_index;
        // count inlier number
        for (int j = 0; j<line_points.size(); j++) {
            double dist = min_config_line.distance(line_points[j]);
            if (dist < inlier_distance_threshold) {
                inlier_index.push_back(j);
                inlier_num++;
            }
        }
        if (inlier_num > best_inlier_num) {
            best_inlier_num = inlier_num;
        }
        else {
            continue;
        }
        
        // fit a better model
        vector<cv::Point3d> inlier_points(inlier_index.size());
        for (int j = 0; j < inlier_index.size(); j++) {
            Eigen::Vector3d p = line_points[inlier_index[j]];
            inlier_points[j] = cv::Point3d(p.x(), p.y(), p.z());
        }
        
        cv::Vec6d line3d;
        cv::fitLine(inlier_points, line3d, CV_DIST_L2, 0, 0.01, 0.01);
        
        Eigen::Vector3d org_point(line3d[3], line3d[4], line3d[5]);
        Eigen::Vector3d direction(line3d[0], line3d[1], line3d[2]);
        best_line = Eigen::ParametrizedLine<double, 3>(org_point, direction);
    }
    
    if (best_inlier_num >= min_inlier_ratio * N) {
        //printf("inlier ratio %f \n", 1.0 * best_inlier_num/N);
        output_line = best_line;
        return true;
    }
    else {
        //printf("inlier ratio %f \n", 1.0 * best_inlier_num/N);
        return false;
    }
    return true;
}

bool fitLine3DRansac(const vector<Eigen::Vector3d > & line_points, Eigen::ParametrizedLine<double, 3> & output_line,
                     vector<unsigned>& output_inlier_index)
{
    if(line_points.size() <= 3) {
        return false;
    }
    const int num_iteration = 100;
    const int N = (int)line_points.size();
    const double inlier_distance_threshold = 0.05;
    double min_inlier_ratio = 0.5;
    int best_inlier_num = 0;
    
    Eigen::ParametrizedLine<double, 3> best_line;
    for (int i = 0; i<num_iteration; i++) {
        // randomly pick 2 points and fit a line
        int k1 = 0;
        int k2 = 0;
        
        do{
            k1 = rand()%N;
            k2 = rand()%N;
        }while (k1 == k2);
        
        Eigen::ParametrizedLine<double, 3> min_config_line = Eigen::ParametrizedLine<double, 3>::Through(line_points[k1], line_points[k2]);
        int inlier_num = 0;
        vector<unsigned int> inlier_index;
        // count inlier number
        for (int j = 0; j<line_points.size(); j++) {
            double dist = min_config_line.distance(line_points[j]);
            if (dist < inlier_distance_threshold) {
                inlier_index.push_back(j);
                inlier_num++;
            }
        }
        if (inlier_num > best_inlier_num) {
            best_inlier_num = inlier_num;
        }
        else {
            continue;
        }
        
        // fit a better model
        vector<cv::Point3d> inlier_points(inlier_index.size());
        for (int j = 0; j < inlier_index.size(); j++) {
            Eigen::Vector3d p = line_points[inlier_index[j]];
            inlier_points[j] = cv::Point3d(p.x(), p.y(), p.z());
        }
        
        cv::Vec6d line3d;
        cv::fitLine(inlier_points, line3d, CV_DIST_L2, 0, 0.01, 0.01);
        
        Eigen::Vector3d org_point(line3d[3], line3d[4], line3d[5]);
        Eigen::Vector3d direction(line3d[0], line3d[1], line3d[2]);
        best_line = Eigen::ParametrizedLine<double, 3>(org_point, direction);
        output_inlier_index = inlier_index;
    }
    
    if (best_inlier_num >= min_inlier_ratio * N) {
        //printf("inlier ratio %f \n", 1.0 * best_inlier_num/N);
        output_line = best_line;
        return true;
    }
    else {
        //printf("inlier ratio %f \n", 1.0 * best_inlier_num/N);
        return false;
    }

    return true;
}

bool fitLine3DRansac(const vector<Eigen::Vector3d > & line_points,
                     Eigen::ParametrizedLine<double, 3> & output_line,
                     std::pair<Eigen::Vector3d, Eigen::Vector3d>& output_line_end_point)
{
    if(line_points.size() <= 3) {
        return false;
    }
    const int num_iteration = 100;
    const int N = (int)line_points.size();
    const double inlier_distance_threshold = 0.05;
    double min_inlier_ratio = 0.5;
    int best_inlier_num = 0;
    
    Eigen::ParametrizedLine<double, 3> best_line;
    for (int i = 0; i<num_iteration; i++) {
        // randomly pick 2 points and fit a line
        int k1 = 0;
        int k2 = 0;
        
        do{
            k1 = rand()%N;
            k2 = rand()%N;
        }while (k1 == k2);
        
        Eigen::ParametrizedLine<double, 3> min_config_line = Eigen::ParametrizedLine<double, 3>::Through(line_points[k1], line_points[k2]);
        int inlier_num = 0;
        vector<int> inlier_index;
        // count inlier number
        for (int j = 0; j<line_points.size(); j++) {
            double dist = min_config_line.distance(line_points[j]);
            if (dist < inlier_distance_threshold) {
                inlier_index.push_back(j);
                inlier_num++;
            }
        }
        if (inlier_num > best_inlier_num) {
            best_inlier_num = inlier_num;
        }
        else {
            continue;
        }
        
        // fit a better model
        vector<cv::Point3d> inlier_points(inlier_index.size());
        for (int j = 0; j < inlier_index.size(); j++) {
            Eigen::Vector3d p = line_points[inlier_index[j]];
            inlier_points[j] = cv::Point3d(p.x(), p.y(), p.z());
        }
        
        cv::Vec6d line3d;
        cv::fitLine(inlier_points, line3d, CV_DIST_L2, 0, 0.01, 0.01);
        
        Eigen::Vector3d org_point(line3d[3], line3d[4], line3d[5]);
        Eigen::Vector3d direction(line3d[0], line3d[1], line3d[2]);
        best_line = Eigen::ParametrizedLine<double, 3>(org_point, direction);
        
        // project first inlier and last inlier to the line
        // @bug the point is not sorted
        Eigen::Vector3d first_p = line_points[inlier_index.front()];
        Eigen::Vector3d last_p  = line_points[inlier_index.back()];
        output_line_end_point.first = best_line.projection(first_p);
        output_line_end_point.second = best_line.projection(last_p);
    }
    
    if (best_inlier_num >= min_inlier_ratio * N) {
        //printf("inlier ratio %f \n", 1.0 * best_inlier_num/N);
        output_line = best_line;
        return true;
    }
    else {
        //printf("inlier ratio %f \n", 1.0 * best_inlier_num/N);
        return false;
    }
    return true;
}

Eigen::Matrix3d JacobQwithA(const Eigen::Vector3d& A, const Eigen::Vector3d& B,
                            const Eigen::Vector3d& P, const Eigen::Matrix3d& lambda)
{
    Eigen::Matrix3d dQdA = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    
    Eigen::Vector3d BminusA = B - A;
    Eigen::Vector3d AminusP = A - P;
    double E = BminusA.transpose() * lambda * AminusP;
    double D = BminusA.transpose() * lambda * BminusA;
    assert(D != 0.0);
    Eigen::Vector3d dEdA = lambda * (B + P - 2.0 * A);  // 3 * 1
    Eigen::Vector3d dDdA = 2.0 * lambda * (A - B)  ;   // 3 * 1
    double C = E/D;
    Eigen::Vector3d dCdA = (dEdA * D - E * dDdA)/(D * D);  // 3 * 1
    Eigen::Matrix3d dCdA_BminusA = dCdA * BminusA.transpose();
    assert(dCdA_BminusA.rows() == 3 && dCdA_BminusA.cols() == 3);
    dQdA = (1.0 + C) * I - dCdA_BminusA;
    
    return dQdA;
}

Eigen::Matrix3d JacobQwithB(const Eigen::Vector3d& A, const Eigen::Vector3d& B,
                            const Eigen::Vector3d& P, const Eigen::Matrix3d& lambda)
{
    Eigen::Matrix3d dQdB = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    
    Eigen::Vector3d BminusA = B - A;
    Eigen::Vector3d AminusP = A - P;
    double E = BminusA.transpose() * lambda * AminusP;
    double D = BminusA.transpose() * lambda * BminusA;
    assert(D != 0.0);
    Eigen::Vector3d dEdB = lambda * (A - P);  // 3 * 1
    Eigen::Vector3d dDdB = 2.0 * lambda * (B - A)  ;   // 3 * 1
    double C = E/D;
    Eigen::Vector3d dCdB = (dEdB * D - E * dDdB)/(D * D);
    dQdB = -1.0 * (dCdB * (B - A).transpose() + C * I);
    
    return dQdB;
}

bool fitLuLine3D(const vector<Eigen::Vector3d > & ordered_line_points,
                 const vector<Eigen::Matrix3d> & ordered_point_precision,
                 Eigen::Vector3d& outputA, Eigen::Vector3d& outputB, Eigen::MatrixXd & putput_cov_line)
{
    assert(ordered_line_points.size() == ordered_point_precision.size());
    
    /*
    // for testing
    cout<<"line points: \n";
    for (int i = 0; i<ordered_line_points.size(); i++) {
        cout<<ordered_line_points[i].transpose()<<endl;
    }
    cout<<endl;
     */
    
    
    // 1. fit a 3D line
    Eigen::ParametrizedLine<double, 3> line;
    bool is_fit = fitLine3D(ordered_line_points, line);
    if (!is_fit) {
        return false;
    }
    outputA = line.projection(ordered_line_points.front());
    outputB = line.projection(ordered_line_points.back());
    
    const int N = (int)ordered_line_points.size();
    
    // 2. estimate line covariance matrix
    Eigen::MatrixXd Jw = Eigen::MatrixXd::Zero(3 * N, 6);
    for (int i = 0; i < N; i++) {
        Eigen::Vector3d p = ordered_line_points[i];
        Eigen::Matrix3d dJdA = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d dJdB = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d lambda = ordered_point_precision[i];
        
        // equation 8
        if (i != 0 && i != N - 1) {
            dJdA = - 1.0 * JacobQwithA(outputA, outputB, p, lambda);
            dJdB = - 1.0 * JacobQwithB(outputA, outputB, p, lambda);
        }
        else if (i == 0) {
            dJdA = - 1.0 * Eigen::Matrix3d::Identity();
            dJdB = Eigen::Matrix3d::Zero();
        }
        else if (i == N - 1) {
            dJdA = Eigen::Matrix3d::Zero();
            dJdB = - 1.0 * Eigen::Matrix3d::Identity();
        }
        Jw.block(3 * i, 0, 3, 3) = dJdA;
        Jw.block(3 * i, 3, 3, 3) = dJdB;
    }
    
    // block matrix
    Eigen::MatrixXd precision = Eigen::MatrixXd::Zero(3 * N, 3 * N);
    for (int i = 0; i<N; i++) {
        precision.block(3 * i, 3 * i, 3, 3) = ordered_point_precision[i];
    }
    Eigen::MatrixXd JwJ = Jw.transpose() * precision * Jw;
    assert(JwJ.rows() == 6 && JwJ.cols() == 6);
    
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(JwJ);
    if (!qr.isInvertible()) {
        return false;
    }
    
    putput_cov_line = JwJ.inverse();
    
    return true;
}

bool fitLuLine3D(const vector<Eigen::Vector3d> & ordered_line_points,
                 const vector<Eigen::Matrix3d> & ordered_point_covariance,
                 Eigen::Vector3d& A, Eigen::Vector3d& B, Eigen::MatrixXd & cov_line,
                 bool check_invert)
{
    vector<Eigen::Vector3d> points;
    vector<Eigen::Matrix3d> ordered_point_precision;
    for (int i = 0; i<ordered_point_covariance.size(); i++) {
        Eigen::Matrix3d cov = ordered_point_covariance[i];
        Eigen::ColPivHouseholderQR<Eigen::Matrix3d> qr(cov);
        if (qr.isInvertible() && cov.determinant() > 0.0) {  // positive definitive
            Eigen::Matrix3d precision = cov.inverse();
            ordered_point_precision.push_back(precision);
            points.push_back(ordered_line_points[i]);
        }
    }
    
    return fitLuLine3D(points, ordered_point_precision, A, B, cov_line);
}

Eigen::Vector3d JacobD2withA(const Eigen::Vector3d& A, const Eigen::Vector3d& B,
                             const Eigen::Vector3d& C)
{
    /* d^2 = squared_norm( (A - C) X (B -C))  / squared_norm(B - A), may be wrong
    Eigen::Vector3d dydA = Eigen::Vector3d::Identity();
    Eigen::Vector3d AminusC = A - C;
    Eigen::Vector3d BminusC = B - C;
    Eigen::Vector3d F = AminusC.cross(BminusC);
    double D = F.squaredNorm();
    double E = (B - A).squaredNorm();
    assert(E != 0.0);
    Eigen::Vector3d dEdA = 2.0 * (A - B);
    Eigen::Matrix3d dFdA = EigenGeometryUtil::vector2SkewSymmetricMatrix(BminusC).transpose();
    Eigen::Vector3d dDdF = 2.0 * F;
    Eigen::Vector3d dDdA = dFdA.transpose() * dDdF;
    
    dydA = (dDdA * E - D * dEdA)/(E * E);
     */     
    
    Eigen::Vector3d dydA = Eigen::Vector3d::Identity();
    Eigen::Vector3d BminusA = B - A;
    Eigen::Vector3d AminusC = A - C;
    Eigen::Vector3d F = BminusA.cross(AminusC);
    double D = F.squaredNorm();
    double E = (B - A).squaredNorm();
    assert(E != 0.0);
    Eigen::Vector3d dEdA = 2.0 * (A - B);
    Eigen::Matrix3d dFdA = EigenGeometryUtil::vector2SkewSymmetricMatrix(B) +
                           EigenGeometryUtil::vector2SkewSymmetricMatrix(C).transpose();
    Eigen::Vector3d dDdF = 2.0 * F;
    Eigen::Vector3d dDdA = dFdA.transpose() * dDdF;
    
    dydA = (dDdA * E - D * dEdA)/(E * E);
     
   
    return dydA;
}

Eigen::Vector3d JacobD2withB(const Eigen::Vector3d& A, const Eigen::Vector3d& B,
                             const Eigen::Vector3d& C)
{
    /*
    Eigen::Vector3d dydB = Eigen::Vector3d::Identity();
    Eigen::Vector3d AminusC = A - C;
    Eigen::Vector3d BminusC = B - C;
    Eigen::Vector3d F = AminusC.cross(BminusC);
    double D = F.squaredNorm();
    double E = (B - A).squaredNorm();
    assert(E != 0.0);
    Eigen::Vector3d dEdB = 2.0 * (B - A);
    Eigen::Vector3d dDdF = 2.0 * F;
    Eigen::Matrix3d dFdB = EigenGeometryUtil::vector2SkewSymmetricMatrix(AminusC);
    Eigen::Vector3d dDdB = dFdB.transpose() * dDdF;
    dydB = (dDdB * E - D * dEdB)/(E * E);
     */
    
    Eigen::Vector3d dydB = Eigen::Vector3d::Identity();
    Eigen::Vector3d BminusA = B - A;
    Eigen::Vector3d AminusC = A - C;
    Eigen::Vector3d F = BminusA.cross(AminusC);
    double D = F.squaredNorm();
    double E = (B - A).squaredNorm();
    assert(E != 0.0);
    Eigen::Vector3d dEdB = 2.0 * (B - A);
    Eigen::Vector3d dDdF = 2.0 * F;
    Eigen::Matrix3d dFdB = EigenGeometryUtil::vector2SkewSymmetricMatrix(AminusC).transpose();
    Eigen::Vector3d dDdB = dFdB.transpose() * dDdF;
    dydB = (dDdB * E - D * dEdB)/(E * E);
    
    return dydB;
}

bool pointToLineDistanceUncertainty(const Eigen::Vector3d& A, const Eigen::Vector3d& B,
                                    const Eigen::MatrixXd& cov_line,
                                    const Eigen::Vector3d& P,
                                    double& distance,
                                    double& sigma)
{
    assert(cov_line.rows() == 6 && cov_line.cols() == 6);
    
    Eigen::ParametrizedLine<double, 3> line = Eigen::ParametrizedLine<double, 3>::Through(A, B);
    distance = line.distance(P);
    
    Eigen::Vector3d dydA = JacobD2withA(A, B, P); // dydA y = d^2
    Eigen::Vector3d dydB = JacobD2withB(A, B, P); //
    double d_d_dy = 0.5 /(distance + 0.000001);  // 1/2 * 1/y^2
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(1, 6);
    J(0, 0) = dydA.x();
    J(0, 1) = dydA.y();
    J(0, 2) = dydA.z();
    J(0, 3) = dydB.x();
    J(0, 4) = dydB.y();
    J(0, 5) = dydB.z();
    J = d_d_dy * J; // chain rule
    Eigen::MatrixXd cov = J * cov_line * J.transpose(); // error forward propagate
    assert(cov.rows() == 1 && cov.cols() == 1);
    sigma = cov(0, 0);
    return true;
}


void backprojectImagePointTo3Dline(const vector<Eigen::Vector2d>& img_pts,
                                   const vector<Eigen::ParametrizedLine<double, 3> >& camera_lines,
                                   const Eigen::Matrix3d& camera_matrix,
                                   vector<Eigen::Vector3d>& points_on_line)
{
    assert(img_pts.size() == camera_lines.size());
    
    Eigen::Matrix3d invK = camera_matrix.inverse();
    for (int i = 0; i<img_pts.size(); i++) {
        Eigen::Vector3d p(img_pts[i].x(), img_pts[i].y(), 1);
        Eigen::Vector3d q = invK * p;  //  a ray in camera coordinate
        q.normalize(); // as direction
        
        Eigen::ParametrizedLine<double, 3> cur_line = camera_lines[i];
        Eigen::Vector3d o = cur_line.origin();
        Eigen::Vector3d d = cur_line.direction(); // direction
        
        Eigen::Vector3d v1(-q.x(), o.x(), d.x());
        Eigen::Vector3d v2(-q.y(), o.y(), d.y());
        Eigen::Vector3d s = v1.cross(v2);  // s = [-lambda, alpha, beta]
        double alpha = s.y();
        double beta = s.z();
        assert(alpha != 0.0);
        Eigen::Vector3d p_back_proj = o + beta/alpha * d;
        points_on_line.push_back(p_back_proj);
        
        // project to image
        /*
        Eigen::Vector3d p_proj = camera_matrix * p_back_proj;
        p_proj /= p_proj.z();
        Eigen::Vector2d pp(p_proj.x(), p_proj.y());
        double dist_2 = (img_pts[i] - pp).norm();
        cout<<"image location distance is "<<dist_2<<endl;
         */
    }
}

template void shiftLineEndPoints(const Eigen::Vector3d & p1, const Eigen::Vector3d & p2, Eigen::Vector3d & p3, Eigen::Vector3d & p4);
template void shiftLineEndPoints(const Eigen::Vector2d & p1, const Eigen::Vector2d & p2, Eigen::Vector2d & p3, Eigen::Vector2d & p4);


















