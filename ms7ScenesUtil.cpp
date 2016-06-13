//
//  ms7ScenesUtil.cpp
//  RGB_RF
//
//  Created by jimmy on 2016-06-06.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "ms7ScenesUtil.hpp"
#include <iostream>

using cv::Mat;
using std::cout;
using std::endl;

Mat Ms7ScenesUtil::read_pose_7_scenes(const char *file_name)
{
    Mat P = Mat::zeros(4, 4, CV_64F);
    FILE *pf = fopen(file_name, "r");
    assert(pf);
    for (int row = 0; row<4; row++) {
        for (int col = 0; col<4; col++) {
            double v = 0;
            fscanf(pf, "%lf", &v);
            P.at<double>(row, col) = v;
        }
    }
    fclose(pf);
    //    cout<<"pose is "<<P<<endl;
    return P;
}

// return CV_64F
Mat Ms7ScenesUtil::camera_depth_to_world_depth(const cv::Mat & camera_depth_img, const cv::Mat & pose)
{
    const int width  = camera_depth_img.cols;
    const int height = camera_depth_img.rows;
    Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = 585.0;
    K.at<double>(1, 1) = 585.0;
    K.at<double>(0, 2) = 320.0;
    K.at<double>(1, 2) = 240.0;
    
    Mat inv_K = K.inv();
    
    cv::Mat world_depth_img = cv::Mat::zeros(height, width, CV_64F);
    Mat loc_img = cv::Mat::zeros(3, 1, CV_64F);
    Mat loc_camera_h = cv::Mat::zeros(4, 1, CV_64F); // homography coordinate
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            double camera_depth = camera_depth_img.at<double>(r, c)/1000.0;
            if ((int)camera_depth == 65535) {
                // invalid depth
                continue;
            }
            loc_img.at<double>(0, 0) = c;
            loc_img.at<double>(1, 0) = r;
            loc_img.at<double>(2, 0) = 1.0;
            Mat loc_camera = inv_K * loc_img;
            double z = loc_camera.at<double>(2, 0);
            double scale = camera_depth/z;
            loc_camera_h.at<double>(0, 0) = loc_camera.at<double>(0, 0) * scale;
            loc_camera_h.at<double>(1, 0) = loc_camera.at<double>(1, 0) * scale;
            loc_camera_h.at<double>(2, 0) = loc_camera.at<double>(2, 0) * scale;
            loc_camera_h.at<double>(3, 0) = 1.0;
            
            Mat x_world = pose * loc_camera_h;
            x_world /= x_world.at<double>(3, 0);
            world_depth_img.at<double>(r, c) = x_world.at<double>(2, 0); // save depth in world coordinate
        }
    }
    return world_depth_img;
}

cv::Mat Ms7ScenesUtil::camera_depth_to_world_coordinate(const cv::Mat & camera_depth_img, const cv::Mat & camera_to_world_pose)
{
    const int width  = camera_depth_img.cols;
    const int height = camera_depth_img.rows;
    Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = 585.0;
    K.at<double>(1, 1) = 585.0;
    K.at<double>(0, 2) = 320.0;
    K.at<double>(1, 2) = 240.0;
    
    Mat inv_K = K.inv();
    
    cv::Mat world_coordinate_img = cv::Mat::zeros(height, width, CV_64FC3);
    Mat loc_img = cv::Mat::zeros(3, 1, CV_64F);
    Mat loc_camera_h = cv::Mat::zeros(4, 1, CV_64F); // homography coordinate
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            double camera_depth = camera_depth_img.at<double>(r, c)/1000.0; // to meter
            if ((int)camera_depth == 65535 || camera_depth < 0.001) {
                // invalid depth
                //printf("invalid depth %lf\n", camera_depth);
                continue;
            }
            loc_img.at<double>(0, 0) = c;
            loc_img.at<double>(1, 0) = r;
            loc_img.at<double>(2, 0) = 1.0;
            Mat loc_camera = inv_K * loc_img;
            double z = loc_camera.at<double>(2, 0);
            double scale = camera_depth/z;
            loc_camera_h.at<double>(0, 0) = loc_camera.at<double>(0, 0) * scale;
            loc_camera_h.at<double>(1, 0) = loc_camera.at<double>(1, 0) * scale;
            loc_camera_h.at<double>(2, 0) = loc_camera.at<double>(2, 0) * scale;
            loc_camera_h.at<double>(3, 0) = 1.0;
            
            Mat x_world = camera_to_world_pose * loc_camera_h;
            x_world /= x_world.at<double>(3, 0);
            world_coordinate_img.at<cv::Vec3d>(r, c)[0] = x_world.at<double>(0, 0);
            world_coordinate_img.at<cv::Vec3d>(r, c)[1] = x_world.at<double>(1, 0);
            world_coordinate_img.at<cv::Vec3d>(r, c)[2] = x_world.at<double>(2, 0);
        }
    }
    //world_coordinate_img /= 1000.0;
    return world_coordinate_img;
}

cv::Mat Ms7ScenesUtil::camera_depth_to_world_coordinate(const cv::Mat & camera_depth_img,
                                                           const cv::Mat & camera_to_world_pose,
                                                           cv::Mat & mask)
{
    assert(camera_depth_img.type() == CV_64FC1);
    const int width  = camera_depth_img.cols;
    const int height = camera_depth_img.rows;
    Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = 585.0;
    K.at<double>(1, 1) = 585.0;
    K.at<double>(0, 2) = 320.0;
    K.at<double>(1, 2) = 240.0;
    
    Mat inv_K = K.inv();
    
    //cout<<"invet K is "<<inv_K<<endl;
    
    cv::Mat world_coordinate_img = cv::Mat::zeros(height, width, CV_64FC3);
    Mat loc_img = cv::Mat::zeros(3, 1, CV_64F);
    Mat loc_camera_h = cv::Mat::zeros(4, 1, CV_64F); // homography coordinate
    mask = cv::Mat::ones(height, width, CV_8UC1);
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            double camera_depth = camera_depth_img.at<double>(r, c)/1000.0; // to meter
            if (camera_depth == 65.535 || camera_depth < 0.1 || camera_depth > 10.0) {
                // invalid depth
                //printf("invalid depth %lf\n", camera_depth);
                mask.at<unsigned char>(r, c) = 0;
                continue;
            }
            loc_img.at<double>(0, 0) = c;
            loc_img.at<double>(1, 0) = r;
            loc_img.at<double>(2, 0) = 1.0;
            Mat loc_camera = inv_K * loc_img;
            double local_z = loc_camera.at<double>(2, 0);
            double scale = camera_depth/local_z;
            //cout<<"scale is "<<scale<<endl;
            loc_camera_h.at<double>(0, 0) = loc_camera.at<double>(0, 0) * scale;
            loc_camera_h.at<double>(1, 0) = loc_camera.at<double>(1, 0) * scale;
            loc_camera_h.at<double>(2, 0) = loc_camera.at<double>(2, 0) * scale;
            loc_camera_h.at<double>(3, 0) = 1.0;
            
            Mat x_world = camera_to_world_pose * loc_camera_h;
            x_world /= x_world.at<double>(3, 0);
            world_coordinate_img.at<cv::Vec3d>(r, c)[0] = x_world.at<double>(0, 0);
            world_coordinate_img.at<cv::Vec3d>(r, c)[1] = x_world.at<double>(1, 0);
            world_coordinate_img.at<cv::Vec3d>(r, c)[2] = x_world.at<double>(2, 0);
        }
    }
    return world_coordinate_img;
}

cv::Mat
Ms7ScenesUtil::camera_depth_to_camera_coordinate(const cv::Mat & camera_depth_img,                                                
                                                 cv::Mat & mask)
{
    assert(camera_depth_img.type() == CV_64FC1);
    
    const int width  = camera_depth_img.cols;
    const int height = camera_depth_img.rows;
    Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = 585.0;
    K.at<double>(1, 1) = 585.0;
    K.at<double>(0, 2) = 320.0;
    K.at<double>(1, 2) = 240.0;
    
    Mat inv_K = K.inv();
    
    //cout<<"invet K is "<<inv_K<<endl;
    
    cv::Mat camera_coordinate_img = cv::Mat::zeros(height, width, CV_64FC3);
    Mat loc_img = cv::Mat::zeros(3, 1, CV_64F);
    Mat loc_camera_h = cv::Mat::zeros(4, 1, CV_64F); // homography coordinate
    mask = cv::Mat::ones(height, width, CV_8UC1);
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            double camera_depth = camera_depth_img.at<double>(r, c)/1000.0; // to meter
            if (camera_depth == 65.535 || camera_depth < 0.1 || camera_depth > 10.0) {
                // invalid depth
                //printf("invalid depth %lf\n", camera_depth);
                mask.at<unsigned char>(r, c) = 0;
                continue;
            }
            loc_img.at<double>(0, 0) = c;
            loc_img.at<double>(1, 0) = r;
            loc_img.at<double>(2, 0) = 1.0;
            Mat loc_camera = inv_K * loc_img;
            double local_z = loc_camera.at<double>(2, 0);
            double scale = camera_depth/local_z;
            //cout<<"scale is "<<scale<<endl;
            camera_coordinate_img.at<cv::Vec3d>(r, c)[0] = loc_camera.at<double>(0, 0) * scale;
            camera_coordinate_img.at<cv::Vec3d>(r, c)[1] = loc_camera.at<double>(1, 0) * scale;
            camera_coordinate_img.at<cv::Vec3d>(r, c)[2] = loc_camera.at<double>(2, 0) * scale;
           
        }
    }
    return camera_coordinate_img;
}



bool Ms7ScenesUtil::load_prediction_result(const char *file_name, string & rgb_img_file, string & depth_img_file, string & camera_pose_file,
                                              vector<cv::Point2d> & img_pts,
                                              vector<cv::Point3d> & wld_pts_pred,
                                              vector<cv::Point3d> & wld_pts_gt)
{
    assert(file_name);
    FILE *pf = fopen(file_name, "r");
    if (!pf) {
        printf("Error, can not read from %s\n", file_name);
        return false;
    }
    
    {
        char buf[1024] = {NULL};
        fscanf(pf, "%s", buf);
        rgb_img_file = string(buf);
    }
    
    {
        char buf[1024] = {NULL};
        fscanf(pf, "%s", buf);
        depth_img_file = string(buf);
    }
    
    {
        char buf[1024] = {NULL};
        fscanf(pf, "%s", buf);
        camera_pose_file = string(buf);
    }
    
    {
        char dummy_buf[1024] = {NULL};
        fgets(dummy_buf, sizeof(dummy_buf), pf);
        printf("%s\n", dummy_buf);
    }
    
    while (1) {
        double val[8] = {0.0};
        int ret = fscanf(pf, "%lf %lf %lf %lf %lf %lf %lf %lf", &val[0], &val[1],
                         &val[2], &val[3], &val[4],
                         &val[5], &val[6], &val[7]);
        if (ret != 8) {
            break;
        }
        // filter out zero points
        img_pts.push_back(cv::Point2f(val[0], val[1]));
        wld_pts_pred.push_back(cv::Point3f(val[2], val[3], val[4]));
        wld_pts_gt.push_back(cv::Point3f(val[5], val[6], val[7]));
    }
    fclose(pf);
    printf("read %lu prediction and ground truth points.\n", wld_pts_gt.size());
    
    return true;
}

bool Ms7ScenesUtil::load_prediction_result_with_color(const char *file_name,
                                                         string & rgb_img_file,
                                                         string & depth_img_file,
                                                         string & camera_pose_file,
                                                         vector<cv::Point2d> & img_pts,
                                                         vector<cv::Point3d> & wld_pts_pred,
                                                         vector<cv::Point3d> & wld_pts_gt,
                                                         vector<cv::Vec3d> & color_pred,
                                                         vector<cv::Vec3d> & color_sample)
{
    assert(file_name);
    FILE *pf = fopen(file_name, "r");
    if (!pf) {
        printf("Error, can not read from %s\n", file_name);
        return false;
    }
    
    {
        char buf[1024] = {NULL};
        fscanf(pf, "%s", buf);
        rgb_img_file = string(buf);
    }
    
    {
        char buf[1024] = {NULL};
        fscanf(pf, "%s", buf);
        depth_img_file = string(buf);
    }
    
    {
        char buf[1024] = {NULL};
        fscanf(pf, "%s\n", buf);   // remove the last \n
        camera_pose_file = string(buf);
    }
    
    {
        char dummy_buf[1024] = {NULL};
        fgets(dummy_buf, sizeof(dummy_buf), pf);
        printf("%s\n", dummy_buf);
    }
    
    while (1) {
        double val[8] = {0.0};
        int ret = fscanf(pf, "%lf %lf %lf %lf %lf %lf %lf %lf", &val[0], &val[1],
                         &val[2], &val[3], &val[4],
                         &val[5], &val[6], &val[7]);
        if (ret != 8) {
            break;
        }
        
        // 2D , 3D position
        img_pts.push_back(cv::Point2d(val[0], val[1]));
        wld_pts_pred.push_back(cv::Point3d(val[2], val[3], val[4]));
        wld_pts_gt.push_back(cv::Point3d(val[5], val[6], val[7]));
        
        double val2[6] = {0.0};
        ret = fscanf(pf, "%lf %lf %lf %lf %lf %lf",
                     &val2[0], &val2[1], &val2[2],
                     &val2[3], &val2[4], &val2[5]);
        if (ret != 6) {
            break;
        }
        color_pred.push_back(cv::Vec3d(val2[0], val2[1], val2[2]));
        color_sample.push_back(cv::Vec3d(val2[3], val2[4], val2[5]));
        assert(img_pts.size() == color_pred.size());
    }
    fclose(pf);
    printf("read %lu prediction and ground truth points.\n", wld_pts_gt.size());
    return true;
}

bool Ms7ScenesUtil::load_estimated_camera_pose(const char *file_name,
                                                  string & rgb_img_file,
                                                  string & depth_img_file,
                                                  string & camera_pose_file,
                                                  cv::Mat & estimated_pose)
{
    assert(file_name);
    FILE *pf = fopen(file_name, "r");
    if (!pf) {
        printf("Error, can not read from %s\n", file_name);
        return false;
    }
    
    {
        char buf[1024] = {NULL};
        fscanf(pf, "%s", buf);
        rgb_img_file = string(buf);
    }
    
    {
        char buf[1024] = {NULL};
        fscanf(pf, "%s", buf);
        depth_img_file = string(buf);
    }
    
    {
        char buf[1024] = {NULL};
        fscanf(pf, "%s\n", buf);   // remove the last \n
        camera_pose_file = string(buf);
    }
    
    estimated_pose = cv::Mat::eye(4, 4, CV_64FC1);
    
    for (int r = 0; r<4; r++) {
        for (int c = 0; c<4; c++) {
            double val = 0.0;
            int ret = fscanf(pf, "%lf", &val);
            assert(ret == 1);
            estimated_pose.at<double>(r, c) = val;
        }
    }
    
    fclose(pf);
    return true;
}

vector<string> Ms7ScenesUtil::read_file_names(const char *file_name)
{
    vector<string> file_names;
    FILE *pf = fopen(file_name, "r");
    assert(pf);
    while (1) {
        char line[1024] = {NULL};
        int ret = fscanf(pf, "%s", line);
        if (ret != 1) {
            break;
        }
        file_names.push_back(string(line));
    }
    printf("read %lu lines\n", file_names.size());
    fclose(pf);
    return file_names;
}
