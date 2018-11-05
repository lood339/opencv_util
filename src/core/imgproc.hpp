//
//  cvx_imgproc.hpp
//  RGB_RF
//
//  Created by jimmy on 2016-06-02.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef cvxImgProc_cpp
#define cvxImgProc_cpp

#include <stdio.h>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/features2d.hpp"
//#include <opencv2/line_descriptor.hpp>

using std::vector;
using cv::Mat;

class CvxImgProc
{
public:
    // | grad_x | + | grad_y |
    // grad: CV_8UC1
    static void imageGradient(const Mat & color_img, Mat & grad);
    
    // gradient orientation in [0, 2 * pi)
    static Mat gradientOrientation(const Mat & img, const int gradMagThreshold = 0);    
    
    
    // centroid orientation (as in ORB) https://en.wikipedia.org/wiki/Image_moment
    // ouput: angles atan2(m01, m10)
    static void centroidOrientation(const Mat & img, const vector<cv::Point2d> & pts, const int patchSize,
                                    vector<float> & angles);
    
    // patchSize:  patch size to calcualte orientation
    // smoothSize: Gaussian kernal size in pixel
    // smooth orientation using Gaussian filter
    static void centroidOrientation(const Mat & img, const int patchSize, const int smoothSize, Mat & orientation);
    
    // put patches into a single image, patch has same size
    // rowNum: how many patches in a column
    static Mat groupPatches(const vector<cv::Mat> & patches, int colNum);
    
    // left side of line should be brighter than right side area
    static bool estimateLineOrientation(const Mat& gry_img, const cv::Point2d & startPoint,
                                        const cv::Point2d & endPoint,const int line_width);    
    
};

namespace cvx {
    // adaptive non-maximal suppression from "Multi-image matching using multi-scale oriented patches". CVPR 2015
    // corner strength is measured by haris corner
    // qualityLevel: set a smaller value (e.g. 0.0001) as we may need to find weak corner in texture-less areas
    // useHarrisDetector: not used
    // k: harris corner parameter, 0.04
    void goodFeaturesToTrack( cv::InputArray image, cv::OutputArray corners,
                             int maxCorners, double qualityLevel, double minDistance,
                             cv::InputArray mask = cv::noArray(), int blockSize = 3,
                             bool useHarrisDetector = false, double k = 0.04 );
    
    // line segment center point tracking by "Distance transforms of sampled functions"
    // src: edge end points on the source image, x1, y1, x2, y2
    // dst: edge end poitns on the destination image
    // centers: Ouput, tracked center point on the destination image, same size as src
    // image_size: size of image
    // center_point_patch_distance: distance from a patch to the destination image
    // dist_map: edge distance map of destination image, CV_32FC1, in or output
    // search_length: search distance along the line direction
    // block_size: size of block used in computing the distance
    void trackLineSegmentCenter(const vector<cv::Vec4f>& src,
                                const vector<cv::Vec4f>& dst,
                                vector<cv::Vec2f>& centers,
                                const cv::Size& image_size,
                                cv::OutputArray center_point_patch_distance = cv::noArray(),
                                cv::InputOutputArray dist_map = cv::noArray(),
                                double search_length = 10,
                                int block_size = 25,
                                bool given_dist_map = false);  
    
    
    
}


#endif /* cvx_imgproc_cpp */
