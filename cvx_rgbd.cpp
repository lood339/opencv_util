//
//  cvx_rgbd.cpp
//  PointLineReloc
//
//  Created by jimmy on 2017-03-30.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "cvx_rgbd.h"
#include "lsd_line_segment.h"
#include "cvx_line.h"
#include <iostream>
#include "vgl_algo.h"
#include <opencv2/photo.hpp>

using std::cout;
using std::endl;

bool CvxRGBD::cameraDepthToWorldCoordinate(const cv::Mat & camera_depth_img,
                                  const cv::Mat & camera_to_world_pose,
                                  const cv::Mat & calibration_matrix,
                                  const double depth_factor,
                                  const double min_depth,
                                  const double max_depth,
                                  cv::Mat & camera_coordinate,
                                  cv::Mat & world_coordinate,
                                  cv::Mat & mask)
{
    assert(camera_depth_img.type() == CV_64FC1);
    assert(camera_to_world_pose.type() == CV_64FC1);
    assert(calibration_matrix.type() == CV_64FC1);
    assert(min_depth < max_depth);
    assert(min_depth >= 0.0);
    
    const int width  = camera_depth_img.cols;
    const int height = camera_depth_img.rows;
    cv::Mat inv_K = calibration_matrix.inv();
    
    cv::Mat world_coordinate_img = cv::Mat::zeros(height, width, CV_64FC3);
    cv::Mat loc_img = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat loc_camera_h = cv::Mat::zeros(4, 1, CV_64F); // homography coordinate
    mask = cv::Mat::ones(height, width, CV_8UC1);
    camera_coordinate = cv::Mat::zeros(height, width, CV_64FC3);
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            double camera_depth = camera_depth_img.at<double>(r, c)/depth_factor; // to meter
            if (camera_depth < min_depth || camera_depth > max_depth ) {
                // invalid depth
                //printf("invalid depth %lf\n", camera_depth);
                mask.at<unsigned char>(r, c) = 0;
                continue;
            }
            loc_img.at<double>(0, 0) = c;
            loc_img.at<double>(1, 0) = r;
            loc_img.at<double>(2, 0) = 1.0;
            cv::Mat loc_camera = inv_K * loc_img;
            double local_z = loc_camera.at<double>(2, 0);
            double scale = camera_depth/local_z;
            //cout<<"scale is "<<scale<<endl;
            loc_camera_h.at<double>(0, 0) = loc_camera.at<double>(0, 0) * scale;
            loc_camera_h.at<double>(1, 0) = loc_camera.at<double>(1, 0) * scale;
            loc_camera_h.at<double>(2, 0) = loc_camera.at<double>(2, 0) * scale;
            loc_camera_h.at<double>(3, 0) = 1.0;
            
            /*
            if(rand()% 120 == 0)
            {
                cv::Mat cam_pt = cv::Mat::zeros(3, 1, CV_64FC1);
                cam_pt.at<double>(0, 0) = loc_camera_h.at<double>(0, 0);
                cam_pt.at<double>(1, 0) = loc_camera_h.at<double>(1, 0);
                cam_pt.at<double>(2, 0) = loc_camera_h.at<double>(2, 0);
                
                cv::Mat temp = calibration_matrix*cam_pt;
                temp /= temp.at<double>(2, 0);
                cout<<"reprojected points location: "<<temp.t()<<endl;
                cout<<"image              location: "<<c<<" "<<r<<endl<<endl;
            }
             */
            
            // the x, y, z in camera coordininate
            camera_coordinate.at<cv::Vec3d>(r,c)[0] = loc_camera_h.at<double>(0, 0);
            camera_coordinate.at<cv::Vec3d>(r,c)[1] = loc_camera_h.at<double>(1, 0);
            camera_coordinate.at<cv::Vec3d>(r,c)[2] = loc_camera_h.at<double>(2, 0);
            
            cv::Mat x_world = camera_to_world_pose * loc_camera_h;
            x_world /= x_world.at<double>(3, 0);
            world_coordinate_img.at<cv::Vec3d>(r, c)[0] = x_world.at<double>(0, 0);
            world_coordinate_img.at<cv::Vec3d>(r, c)[1] = x_world.at<double>(1, 0);
            world_coordinate_img.at<cv::Vec3d>(r, c)[2] = x_world.at<double>(2, 0);
        }
    }
    return true;
}

bool CvxRGBD::cameraDepthToCameraCoordinate(const cv::Mat & camera_depth_img,
                                   const cv::Mat & calibration_matrix,
                                   const double depth_factor,
                                   const double min_depth,
                                   const double max_depth,
                                   cv::Mat & camera_coordinate,
                                   cv::Mat & mask)
{
    assert(camera_depth_img.type() == CV_64FC1);
    assert(calibration_matrix.type() == CV_64FC1);
    assert(min_depth < max_depth);
    assert(min_depth >= 0.0);
    
    const int width  = camera_depth_img.cols;
    const int height = camera_depth_img.rows;
    cv::Mat inv_K = calibration_matrix.inv();
    
    cv::Mat world_coordinate_img = cv::Mat::zeros(height, width, CV_64FC3);
    cv::Mat loc_img = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat loc_camera_h = cv::Mat::zeros(4, 1, CV_64F); // homography coordinate
    mask = cv::Mat::ones(height, width, CV_8UC1);
    camera_coordinate = cv::Mat::zeros(height, width, CV_64FC3);
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            double camera_depth = camera_depth_img.at<double>(r, c)/depth_factor; // to meter
            if (camera_depth < min_depth || camera_depth > max_depth ) {
                mask.at<unsigned char>(r, c) = 0;
                continue;
            }
            loc_img.at<double>(0, 0) = c;
            loc_img.at<double>(1, 0) = r;
            loc_img.at<double>(2, 0) = 1.0;
            cv::Mat loc_camera = inv_K * loc_img;
            double local_z = loc_camera.at<double>(2, 0);
            double scale = camera_depth/local_z;
            //cout<<"scale is "<<scale<<endl;
            loc_camera_h.at<double>(0, 0) = loc_camera.at<double>(0, 0) * scale;
            loc_camera_h.at<double>(1, 0) = loc_camera.at<double>(1, 0) * scale;
            loc_camera_h.at<double>(2, 0) = loc_camera.at<double>(2, 0) * scale;
            loc_camera_h.at<double>(3, 0) = 1.0;
            
            // the x, y, z in camera coordininate
            camera_coordinate.at<cv::Vec3d>(r,c)[0] = loc_camera_h.at<double>(0, 0);
            camera_coordinate.at<cv::Vec3d>(r,c)[1] = loc_camera_h.at<double>(1, 0);
            camera_coordinate.at<cv::Vec3d>(r,c)[2] = loc_camera_h.at<double>(2, 0);
        }
    }

    return true;
}

static Eigen::Vector2d cameraToImageProjection(const Eigen::Matrix3d & K, const Eigen::Vector3d & p)
{
    Eigen::Vector3d q = K * p;
    q /= q.z();
    return Eigen::Vector2d(q.x(), q.y());
}

bool CvxRGBD::detect3DLines(const cv::Mat & color_img,
                            const cv::Mat & depth_img,
                            const cv::Mat & mask,
                            const cv::Mat & calibration_matrix,
                            const cv::Mat & camera_coordinate,
                            const RGBDLineParameter & line_param,
                            vector<RGBDLineSegment> & line_segments)
{
    assert(color_img.type() == CV_8UC3);
    assert(depth_img.type() == CV_64FC1);
    assert(mask.type() == CV_8UC1);
    assert(calibration_matrix.type() == CV_64FC1);
    assert(camera_coordinate.type() == CV_64FC3);
    
    // calibration matrix
    Eigen::Matrix3d k_matrix;
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            k_matrix(r, c) = calibration_matrix.at<double>(r, c);
        }
    }
    
    // 1. detect 2D lines
    std::vector<LSD::LSDLineSegment2D> line2d;
    cv::Mat gray_img;
    cv::cvtColor(color_img, gray_img, CV_BGR2GRAY);
    gray_img.convertTo(gray_img, CV_64FC1);
    if (!gray_img.isContinuous()) {
        gray_img = gray_img.clone();
    }
    assert(gray_img.isContinuous());
    const int width = gray_img.cols;
    const int height = gray_img.rows;
   
    LSD::detectLines((double *) gray_img.data, gray_img.cols, gray_img.rows, line2d);
    
    if (/* DISABLES CODE */ (0))
    {
        cv::Mat img = color_img.clone();
        for (int i = 0; i<line2d.size(); i++) {
            cv::Point p1(line2d[i].x1, line2d[i].y1);
            cv::Point p2(line2d[i].x2, line2d[i].y2);
            cv::line(img, p1, p2, cv::Scalar(255, 0, 128), 1);
        }
        cv::imshow("lines", img);
        cv::waitKey();
    }
    
 //   RGBDLineParameter line_param;
    double min_length = line_param.min_length;
    double min_camera_points = line_param.min_camera_points;
    double inlier_point_distance = line_param.inlier_point_threshold;
    double minimum_ratio = line_param.inlier_ratio;
    double line_area_width = line_param.line_area_width;
    double brightness_contradict_threshold = line_param.brightness_contradict_threshold;
    
    // 2. estimate 3D line and its end points in camera coordinate
    for (int i = 0; i<line2d.size(); i++) {
        const Eigen::Vector2d p1(line2d[i].x1, line2d[i].y1);
        const Eigen::Vector2d p2(line2d[i].x2, line2d[i].y2);
        double dist = (p1-p2).norm();
        if (dist < min_length) {  // line segment is too short
            continue;
        }
        
        // pixel location in image space
        vector<Eigen::Vector2d> image_line_points;
        getLinePixels(p1, p2, image_line_points);
        
        // points in camere coordinate
        vector<Eigen::Vector3d> camera_line_points;
        for (int j = 0; j<image_line_points.size(); j++) {
            int x = image_line_points[j].x();
            int y = image_line_points[j].y();
            if (x >= 0 && x < width && y >= 0 && y < height) {
                if (mask.at<unsigned char>(y, x) != 0) {
                    cv::Vec3d p_c = camera_coordinate.at<cv::Vec3d>(y, x);
                    camera_line_points.push_back(Eigen::Vector3d(p_c[0], p_c[1], p_c[2]));
                }
            }
        }
        if (camera_line_points.size() < min_camera_points) {  // too few 3D sampling points
            continue;
        }
        
        // fit a 3D line from a group of 3D points
        Eigen::ParametrizedLine<double, 3> line3d;
        fitLine3D(camera_line_points, line3d);
        
        int num_inliers = 0;
        Eigen::Vector3d first_projected_inlier;
        Eigen::Vector3d last_projected_inlier;
        for (int j = 0; j<camera_line_points.size(); j++) {
            //double dist = line3d.distance(camera_line_points[j]);
            Eigen::Vector3d q = line3d.projection(camera_line_points[j]);
            double dist = (q - camera_line_points[j]).norm();
            if (dist < inlier_point_distance) {
                if (num_inliers == 0) {
                    first_projected_inlier = q;
                }
                last_projected_inlier = q;
                num_inliers++;
            }
        }
        double ratio = 1.0 * num_inliers/camera_line_points.size();
        if (ratio < minimum_ratio) {
            continue;
        }
        
        // project points from camera coordiante to image coordinate
        Eigen::Vector2d p3 = cameraToImageProjection(k_matrix, first_projected_inlier);
        Eigen::Vector2d p4 = cameraToImageProjection(k_matrix, last_projected_inlier);
        
        Eigen::ParametrizedLine<double, 2> line2d;
        line2d = line2d.Through(p1, p2);
        double dist1 = line2d.distance(p3);
        double dist2 = line2d.distance(p4);
        //            cout<<"distances are: "<<dist1<<" "<<dist2<<endl;
        if (dist1 > 1.0 || dist2 > 1.0) {  // fixed parameter in image space
            continue;
        }
        
        
        
        // detect direction
        vector<vgl_point_2d<double> > left_side_pts;
        vector<vgl_point_2d<double> > right_side_pts;
        {
            vgl_point_2d<double> p5(p3.x(), p3.y());
            vgl_point_2d<double> p6(p4.x(), p4.y());
            vgl_line_segment_2d<double> seg(p5, p6);
            VglAlgo::pixelAlongLineSegment(vgl_line_segment_2d<double>(p5, p6), line_area_width, width, height, left_side_pts, right_side_pts);
        }
        if (left_side_pts.size() < 50 || right_side_pts.size() < 50) {
            continue;
        }
        
        double left_side_brightness = 0.0;
        double right_side_brightness = 0.0;
        for (int j = 0; j<left_side_pts.size(); j++) {
            int x = left_side_pts[j].x();
            int y = left_side_pts[j].y();
            left_side_brightness += gray_img.at<unsigned char>(y, x);
        }
        left_side_brightness /= left_side_pts.size();
        for (int j = 0; j<right_side_pts.size(); j++) {
            int x = right_side_pts[j].x();
            int y = right_side_pts[j].y();
            right_side_brightness += gray_img.at<unsigned char>(y, x);
        }
        right_side_brightness /= right_side_pts.size();
        
        double dif = fabs(right_side_brightness - left_side_brightness);
        double bright_contradict_ratio = dif/std::max(right_side_brightness, left_side_brightness);
        if (bright_contradict_ratio < brightness_contradict_threshold) {
            continue;
        }
    //    printf("brightness contradict ratio: %lf\n", bright_contradict_ratio);
        
        
        if (left_side_brightness < right_side_brightness) {
            std::swap(p3, p4);
            std::swap(first_projected_inlier, last_projected_inlier);
        }
        
        RGBDLineSegment segment;
        segment.img_p1 = line2d.projection(p3);
        segment.img_p2 = line2d.projection(p4);
        segment.cam_p1 = first_projected_inlier;
        segment.cam_p2 = last_projected_inlier;
        
        line_segments.push_back(segment);
    }
    
    //printf("find %lu line segment\n", line_segments.size());
    
    return true;
}

bool CvxRGBD::depthInpaint(cv::Mat & depth_img,
                           const unsigned char no_depth_mask)
{
    assert(depth_img.type() == CV_16UC1);
    
    int height = depth_img.rows;
    int width  = depth_img.cols;
    
    // depth image with 8UC format
    cv::Mat depth(height, width, CV_8UC1);
    double min_v, max_v;
    cv::minMaxLoc(depth_img, &min_v, &max_v);
    
    double scale = 255.0/(max_v + 100.0);
    depth_img.convertTo(depth, CV_8UC1, scale);
    
    cv::Mat temp, temp2;
    
    // 1 step - downsize for performance, use a smaller version of depth image
    cv::Mat small_depth;
    resize(depth, small_depth, cv::Size(), 0.2, 0.2);
    
    // 2 step - inpaint only the masked "unknown" pixels
    cv::inpaint(small_depth, (small_depth == no_depth_mask), temp, 5.0, cv::INPAINT_TELEA);
    
    // 3 step - upscale to original size and replace inpainted regions in original depth image
    cv::resize(temp, temp2, depth.size());
    temp2.copyTo(depth, (depth == no_depth_mask)); // add to the original signal
    
    cv::Mat converted_depth = depth_img.clone();
    depth.convertTo(converted_depth, CV_16UC1, 1.0/scale);
    converted_depth.copyTo(depth_img, (depth_img == no_depth_mask));
    return true;
}
