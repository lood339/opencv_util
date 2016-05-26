//
//  cvx_asift_warp.cpp
//  ASIFT
//
//  Created by jimmy on 2015-10-21.
//  Copyright © 2015 jimmy. All rights reserved.
//

#include "cvx_asift_warp.hpp"
#include <vector>

using namespace std;
using namespace cv;


Mat cvx_asift_warp::warp_image(const Mat & image, double rotate, double tilt, Mat & warped_image)
{
    assert(tilt >= 0.0 && tilt <= 90.0);
    assert(rotate >= -180.0);
    assert(rotate <=  180.0);
    
    // rotate first, then tilt
    Mat rotated_image;
    Mat rot_affine  = cvx_asift_warp::compute_rotation_image(image, rotate, rotated_image);
    Mat tilt_affine = cvx_asift_warp::compute_tilt_image(rotated_image, tilt, warped_image);
    
    // copy 2 * 3 matrix to 3 * 3 matrix with last row 0, 0, 1
    Mat rot_H = Mat::eye(3, 3, CV_64F);
    Mat aux1 = rot_H.colRange(0, 3).rowRange(0, 2);
    rot_affine.copyTo(aux1);
    
    Mat tilt_H = Mat::eye(3, 3, CV_64F);
    Mat aux2 = tilt_H.colRange(0, 3).rowRange(0, 2);
    tilt_affine.copyTo(aux2);
    
    //
    Mat H = tilt_H * rot_H;
    Mat aux3 = H.colRange(0,3).rowRange(0, 2);
    Mat affine = Mat(2, 3, CV_64F, 0.0);
    aux3.copyTo(affine);
    return affine;
}

vector<Vec2d> cvx_asift_warp::affine_warp_points(const vector<Vec2d> & points, const Mat & affine_transform)
{
    assert(affine_transform.rows == 2);
    assert(affine_transform.cols == 3);
    
    vector<Vec2d> warped_points;
    for (int i = 0; i<points.size(); i++) {
        Mat p_mat(3, 1, CV_64F);
        p_mat.at<double>(0, 0) = points[i][0];
        p_mat.at<double>(1, 0) = points[i][1];
        p_mat.at<double>(2, 0) = 1.0;
        Mat q = affine_transform * p_mat;
        assert(q.rows == 2);
        assert(q.cols == 1);
        double x = q.at<double>(0, 0);
        double y = q.at<double>(1, 0);
        warped_points.push_back(Vec2d(x, y));
    }
    return warped_points;
}

static Mat gray_to_rgb(const Mat & image)
{
    Mat ret = Mat(image.rows, image.cols, 3);
    if (image.channels() == 1) {
        vector<Mat> three_channels;
        three_channels.push_back(image.clone());
        three_channels.push_back(image.clone());
        three_channels.push_back(image.clone());
        cv::merge(three_channels, ret);
    }
    else if(image.channels() == 3)
    {
        ret = image.clone();
    }
    else
    {
        assert(0);
    }
    return ret;
}

Mat cvx_asift_warp::draw_match(const Mat & image1, const Mat & image2,
                               const vector<Vec2d> & points1,
                               const vector<Vec2d> & points2)
{
    assert(points1.size() == points2.size());
    
    Mat rgb1 = gray_to_rgb(image1);
    Mat rgb2 = gray_to_rgb(image2);
    
    const int w1 = image1.cols;
    const int h1 = image1.rows;
    const int w2 = image2.cols;
    const int h2 = image2.rows;
    
    const int gap = 0;
    const int w = w1 + gap + w2;
    const int h = std::max(h1, h2);
    
    Mat matched_image = Mat::zeros(h, w, CV_8UC3);
    //cv::imshow("org match image", matched_image);
    
    matched_image.adjustROI(0, h1, 0, w1);
    rgb1.copyTo(matched_image.colRange(0, w1).rowRange(0, h1));
    rgb2.copyTo(matched_image.colRange(w1+gap, w1+w2+gap).rowRange(0, h2));
    
    for (int i = 0; i < points1.size(); i++) {
        cv::Point p1(points1[i][0], points1[i][1]);
        cv::Point p2(points2[i][0] + w1 + gap, points2[i][1]);
        cv::line(matched_image, p1, p2, cv::Scalar(255, 0, 255));
    }
    
    return matched_image;
}


Mat cvx_asift_warp::compute_tilt_image(const Mat & image, const double tilt, Mat & warped_image)
{
    assert(tilt >= 0.0 && tilt <= 90.0);
    
    double tilt_radian = tilt * M_PI/180.0;
    double scale = cos(tilt_radian);
    assert(scale > 0.0);
    
    const int w = image.cols;
    const int h = image.rows;
    
    cv::Size sz(1.0, scale);     // only scale the y direction
    warped_image = image.clone();
    cv::GaussianBlur(warped_image, warped_image, Size(0,0), 0.01);
    cv::resize(warped_image, warped_image, Size(w,h*scale), 1.0, scale, INTER_LINEAR);
    
    // cv::imshow("warpped image", warped_image);
    // cv::imshow("original image",image);
    Mat r = Mat(2, 3, CV_64F, 0.0);
    r.at<double>(0, 0) = 1.0;
    r.at<double>(1, 1) = scale;
    return r;
}

Mat cvx_asift_warp::compute_rotation_image(const Mat & image, const double rotation, Mat & warped_image)
{
    assert(rotation >= -180.0);
    assert(rotation <=  180.0);
    const int w = image.cols;
    const int h = image.rows;
    
    cv::Point2f pt(w/2.0, h/2.0);
    cv::Mat r = cv::getRotationMatrix2D(pt, rotation, 1.0);
    
    vector<Vec2d> corners;
    corners.push_back(Vec2d(0, 0));
    corners.push_back(Vec2d(w, 0));
    corners.push_back(Vec2d(0, h));
    corners.push_back(Vec2d(w, h));
    
    vector<Vec2d> rotated_corners;
    for (int i = 0; i<corners.size(); i++) {
        Mat p_mat(3, 1, CV_64F);
        p_mat.at<double>(0, 0) = corners[i][0];
        p_mat.at<double>(1, 0) = corners[i][1];
        p_mat.at<double>(2, 0) = 1.0;
        Mat rotated_p = r * p_mat;
        assert(rotated_p.rows == 2);
        assert(rotated_p.cols == 1);
        double x = rotated_p.at<double>(0, 0);
        double y = rotated_p.at<double>(1, 0);
        rotated_corners.push_back(Vec2d(x, y));
    }
    
    //  Rect rect = cv::boundingRect(rotated_corners);
    // bounding box of warped corners
    int x_min = INT_MAX;
    int y_min = INT_MAX;
    int x_max = INT_MIN;
    int y_max = INT_MIN;
    for (int i = 0; i<rotated_corners.size(); i++) {
        int x = cvRound(rotated_corners[i][0]);
        int y = cvRound(rotated_corners[i][1]);
        x_min = (x < x_min) ? x: x_min;
        y_min = (y < y_min) ? y: y_min;
        x_max = (x > x_max) ? x: x_max;
        y_max = (y > y_max) ? y: y_max;
    }
    
    // new image size
    cv::Size sz(x_max - x_min, y_max - y_min);
    // translated x and y, so that the the warped image will be inside the image
    double x_translate = -x_min;
    double y_translate = -y_min;
    r.at<double>(0, 2) = r.at<double>(0, 2) + x_translate;
    r.at<double>(1, 2) = r.at<double>(1, 2) + y_translate;
    
    cv::warpAffine(image, warped_image, r, sz);
    
    return r;
}
