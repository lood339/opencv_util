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

using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::ParametrizedLine;

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
    
    {
        double d1 = (p1 - p3).norm();
        double d2 = (p2 - p4).norm();
        //printf("distance before shifting: %lf %lf\n", d1, d2);
    }
    Vector3d p0 = p3;  // temporal copy
    p3 = p0 + gamma * v;
    p4 = p0 + (gamma + dist) * v;
    
    {
        double d1 = (p1 - p3).norm();
        double d2 = (p2 - p4).norm();
        //printf("distance after shifting: %lf %lf\n\n", d1, d2);
    }
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
