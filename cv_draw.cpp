//
//  cv_draw.cpp
//  MAVGoogleImageMatching
//
//  Created by jimmy on 2016-01-11.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "cv_draw.hpp"

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
    for (int i = 0; i<pts1.size(); i++) {
        cv::Point p1(pts1[i].x, pts1[i].y);
        cv::Point p2(pts2[i].x, pts2[i].y + image1.rows + gap);
        cv::line(matches, p1, p2, cv::Scalar(255, 0, 0));
    }    
}