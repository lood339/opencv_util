//
//  cvxIO.hpp
//  RGB_RF
//
//  Created by jimmy on 2016-05-26.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef cvxIO_cpp
#define cvxIO_cpp

#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include <string>
#include <vector>

using std::string;
using std::vector;


class CvxIO
{
public:
    // unit: millimeter
    static bool imread_depth_16bit_to_32f(const char *file, cv::Mat & depth_img);
    // unit: millimeter
    static bool imread_depth_16bit_to_64f(const char *filename, cv::Mat & depth_img);
    static bool imread_rgb_8u(const char *file_name, cv::Mat & rgb_img);
    static bool imread_gray_8u(const char *file_name, cv::Mat & grey_img);
    
    // write depth image as 8u for visualization purpose
    static void imwrite_depth_8u(const char *file, const cv::Mat & depth_img);
    static void imwrite_xyz_to_8urgb(const char *file, const cv::Mat & xyz_img);
    
    // format as double
    static bool save_mat(const char *txtfile, const cv::Mat & mat);
    static bool load_mat(const char *txtfile, cv::Mat & mat);
    
    static bool write_mat(FILE *pf, const cv::Mat & mat);
    static bool read_mat(FILE *pf, cv::Mat & mat);
    
    // dir_name = "/Users/jimmy/*.txt"
    static vector<string> read_files(const char *dir_name);
    
    // in the file: there is a list of strings
    static vector<string> read_file_names(const char *file_name);
    
};


#endif /* cvxIO_cpp */
