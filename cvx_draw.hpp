//
//  cvx_draw.hpp
//  MAVGoogleImageMatching
//
//  Created by jimmy on 2016-01-11.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef cvx_draw_cpp
#define cvx_draw_cpp

// util functions for draw images
#include <stdio.h>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"

using std::vector;
using cv::Mat;

class CvxDraw
{
public:
    /*
     input: image1, image2, RGB color image
            pts1, pts2, points in iamges
     output: matches, vertical image pairs of image1 and image2, pts1 and pts2 area connected by red lines
     sample_num: when the number of pts1 and pts2 are too large, sampling drawing points
     */
    static void draw_match_vertical(const cv::Mat &image1,
                                    const cv::Mat &image2,
                                    const vector< cv::Point2d > & pts1,
                                    const vector< cv::Point2d > & pts2,
                                    cv::Mat & matches,
                                    const int sample_num = 1);
    
    template <class T>
    static void draw_match_vertical_template(const cv::Mat &image1,
                                             const cv::Mat &image2,
                                             const vector< T > & pts1,
                                             const vector< T > & pts2,
                                             cv::Mat & matches,
                                             const int sample_num = 1)
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
        cv::Mat roi(matches, cv::Rect(0, 0, image1.cols, image1.rows));
        image1.copyTo(roi);
        
        roi = matches(cv::Rect(0, image1.rows + gap, image2.cols, image2.rows));
        image2.copyTo(roi);
        
        // draw lines
        for (int i = 0; i<pts1.size(); i += sample_num) {
            cv::Point p1(pts1[i].x, pts1[i].y);
            cv::Point p2(pts2[i].x, pts2[i].y + image1.rows + gap);
            int r = rand()%256;
            int g = rand()%256;
            int b = rand()%256;
            cv::line(matches, p1, p2, cv::Scalar(b, g, r), 1);
        }
    }

    
    
    // draw cross around the point
    static void draw_cross(cv::Mat & image,
                           const vector<cv::Point2f> & pts,
                           const cv::Scalar & color,
                           const int length = 5);
    
    template< class T>
    static void draw_cross_template(cv::Mat & image, const vector<T> & pts, const cv::Scalar & color, const int length = 5, const int thickness = 1)
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
            
            cv::line(image, p1, p2, color, thickness);
            cv::line(image, p3, p4, color, thickness);
        }
    }

    // error_image: output of parameter
    static void draw_reprojection_error(const vector<cv::Point3f> &pt_3d,
                                        const vector<cv::Point2f> &pt_2d,
                                        const cv::Mat & camera_intrinsic_matrix,
                                        const cv::Mat & rotation,
                                        const cv::Mat & translation,
                                        const cv::Mat & distCoeffs,
                                        vector<float> & reproj_errors,
                                        cv::Mat & error_image);
    
    // visualize gradient using hsv color
    // magnitude [0, 255)
    // orientation [0, 2*pi]
    static cv::Mat visualize_gradient(const Mat & magnitude, const Mat & orientation);
                                   
    
   
    
    // rgb CV_8U image, temporal function
    static void copy_rgb_image(const unsigned char * data,
                               int img_width,
                               int img_height,
                               cv::Mat &image);
    
    
    
    static cv::Scalar red();
    static cv::Scalar green();
    static cv::Scalar blue();
    static cv::Scalar yellow();
};






#endif /* cv_draw_cpp */
