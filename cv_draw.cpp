//
//  cv_draw.cpp
//  MAVGoogleImageMatching
//
//  Created by jimmy on 2016-01-11.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "cv_draw.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace::cv;

void CvDraw::draw_match_vertical(const cv::Mat &image1, const cv::Mat &image2,
                                 const vector< cv::Point2d > & pts1,
                                 const vector< cv::Point2d > & pts2,
                                 cv::Mat &matches, const int sample_num)
{
    assert(image1.channels() == 3);
    assert(image2.channels() == 3);
    assert(image1.type() == CV_8UC3);
    assert(image2.type() == CV_8UC3);
    assert(pts1.size() == pts2.size());
    
    int gap = 10;
    int w = std::max(image1.cols, image2.cols);
    int h = image1.rows + image2.rows + gap;
    matches = cv::Mat(h, w, CV_8UC3);
    
    // copy images
    cv::Mat roi(matches, Rect(0, 0, image1.cols, image1.rows));
    image1.copyTo(roi);
    
    roi = matches(Rect(0, image1.rows + gap, image2.cols, image2.rows));
    image2.copyTo(roi);
    
    // draw lines
    for (int i = 0; i<pts1.size(); i += sample_num) {
        cv::Point p1(pts1[i].x, pts1[i].y);
        cv::Point p2(pts2[i].x, pts2[i].y + image1.rows + gap);
        int r = rand()%256;
        int g = rand()%256;
        int b = rand()%256;
        cv::line(matches, p1, p2, cv::Scalar(b, g, r));
    }    
}

void CvDraw::draw_reprojection_error(const vector<cv::Point3f> &pts_3d,
                                     const vector<cv::Point2f> &pts_2d,
                                     const cv::Mat & camera_intrinsic_matrix,
                                     const cv::Mat & rotation,
                                     const cv::Mat & translation,
                                     const cv::Mat & distCoeffs,
                                     vector<float> & reproj_errors,
                                     cv::Mat & error_image)
{
    assert(pts_3d.size() == pts_2d.size());
    
    // project world point to image space
    vector<Point2f> projectedPoints;
    projectedPoints.resize(pts_3d.size());
    cv::projectPoints(Mat(pts_3d), rotation, translation, camera_intrinsic_matrix, distCoeffs, projectedPoints);
    
    CvDraw::draw_cross(error_image, pts_2d, CvDraw::red());
    CvDraw::draw_cross(error_image, projectedPoints, CvDraw::green());
    
    // draw reprojection error
    for (int i = 0; i<pts_2d.size(); i++) {
        double dx = pts_2d[i].x - projectedPoints[i].x;
        double dy = pts_2d[i].y - projectedPoints[i].y;
        double err = sqrt(dx * dx + dy * dy);
        reproj_errors.push_back(err);
        
        cv::Point p1 = cv::Point(pts_2d[i].x, pts_2d[i].y);
        cv::Point p2 = cv::Point(projectedPoints[i].x, projectedPoints[i].y);
        cv::line(error_image, p1, p2, CvDraw::blue());
    }
}

cv::Mat CvDraw::visualize_gradient(const Mat & magnitude, const Mat & orientation)
{
    assert(magnitude.rows == orientation.rows);
    assert(magnitude.cols == orientation.cols);
    assert(magnitude.channels() == 1);
    assert(orientation.channels() == 1);
    
    cv::Mat mag, ori;
    magnitude.convertTo(mag, CV_64FC1);
    orientation.convertTo(ori, CV_64FC1);
    
    int h = mag.rows;
    int w = mag.cols;
    cv::Mat vmat = cv::Mat::zeros(h, w, CV_8UC3);
    for (int y = 0; y<h; y++) {
        for (int x = 0; x<w; x++) {
            //printf("%f\n", ori.at<double>(y, x));
            int h = (ori.at<double>(y, x))/(CV_2PI) * 120;
            int s = mag.at<double>(y, x);
            vmat.at<cv::Vec3b>(y, x)[0] = h;
            vmat.at<cv::Vec3b>(y, x)[1] = s;
            vmat.at<cv::Vec3b>(y, x)[2] = 255;
        }
    }
    cv::cvtColor(vmat, vmat, CV_HSV2BGR);
    return vmat;
}

void CvDraw::draw_cross(cv::Mat & image,
                        const vector<cv::Point2f> & pts,
                        const cv::Scalar & color,
                        const int length)
{
    assert(image.channels() == 3);
    
    for (unsigned int i = 0; i<pts.size(); i++)
    {
        //center point
        int px = pts[i].x;
        int py = pts[i].y;
        
        cv::Point p1, p2, p3, p4;
        
        int h_l = length/2;
        p1 = cv::Point(px - h_l, py);
        p2 = cv::Point(px + h_l, py);
        p3 = cv::Point(px, py - h_l);
        p4 = cv::Point(px, py + h_l);
     
        cv::line(image, p1, p2, color);
        cv::line(image, p3, p4, color);
    }   
}

void CvDraw::copy_rgb_image(const unsigned char * data,
                            int img_width,
                            int img_height,
                            cv::Mat &image)
{
//     Mat(int rows, int cols, int type, void* data, size_t step=AUTO_STEP);
    cv::Mat temp = cv::Mat(img_height, img_width, CV_8UC3, (void*)data);
    image = temp.clone();
}

cv::Scalar CvDraw::red()
{
    return cv::Scalar(0, 0, 255);
    
}

cv::Scalar CvDraw::green()
{
    return cv::Scalar(0, 255, 0);
}

cv::Scalar CvDraw::blue()
{
    return cv::Scalar(255, 0, 0);
}

