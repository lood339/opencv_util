//
//  cvxImgMatch.h
//  RGBD_RF
//
//  Created by jimmy on 2016-09-06.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __RGBD_RF__cvxImgMatch__
#define __RGBD_RF__cvxImgMatch__

// matching two images in pixel level

#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <vector>

using std::vector;

struct SIFTMatchingParameter
{
    
};

class CvxImgMatch
{
public:
    static void SIFTMatching(const cv::Mat & srcImg, const cv::Mat & dstImg,
                             const SIFTMatchingParameter & param, 
                             vector<cv::Point2d> & srcPts, vector<cv::Point2d> & dstPts);
    
    // nearest neighbor matching
    static void NNMatching(const cv::Mat & srcDescriptors, const cv::Mat & dstDescriptors,
                           const vector<cv::Point2d> & srcPts, const vector<cv::Point2d> & dstPts,
                           vector<cv::Point2d> & matchedSrcPts, vector<cv::Point2d> & matchedDstPts);
    
    static void ORBMatching(const cv::Mat & srcImg, const cv::Mat & dstImg,
                            vector<cv::Point2d> & srcPts, vector<cv::Point2d> & dstPts);
    
};


#endif /* defined(__RGBD_RF__cvxImgMatch__) */
