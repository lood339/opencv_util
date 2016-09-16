//
//  cvxWalshHadamard.cpp
//  RGBD_RF
//
//  Created by jimmy on 2016-08-28.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "cvxWalshHadamard.h"
#include "wh.h"

bool CvxWalshHadamard::generateWHFeature(const cv::Mat & image,
                                                   const vector<cv::Point2d> & pts,
                                                   const int patchSize,
                                                   const int kernelNum,                                                   
                                                   vector<Eigen::VectorXf> & features)
{
    assert(image.type() == CV_8UC3);
    assert(patchSize == 4 || patchSize == 8 || patchSize == 32 || patchSize == 64 || patchSize == 128);
    assert(kernelNum <= patchSize * patchSize);
    
    //
    const int row = image.rows;
    const int col = image.cols;
    const int half_size = patchSize/2;
    cv::Mat pad_img;
    cv::copyMakeBorder(image, pad_img, half_size, half_size, half_size, half_size, cv::BORDER_CONSTANT, cv::Scalar(0));
    
    // split rgb to three dimension
    vector<cv::Mat> single_channels;
    cv::split(pad_img, single_channels);
    
    WHSetup *setup = createWHSetup(image.rows, image.cols, patchSize, kernelNum);
    
    Image *pattern = createImageHead(patchSize, patchSize);
    
    // extract sub image
    for (int i = 0; i<pts.size(); i++) {
        int x = pts[i].x;    // original x is the center of the patch, it becomes the top-left of the patch as padding
        int y = pts[i].y;
        assert(x >= 0 && x < col && y >= 0 && y < row);
        
        // test for channel
        Eigen::VectorXf feat(kernelNum * single_channels.size());
        for (int cha = 0; cha < single_channels.size(); cha++) {
            cv::Mat patch;
            single_channels[cha](cv::Rect(x, y, patchSize, patchSize)).copyTo(patch);
            
            assignImageData(pattern, patch.data);
            setPatternImage(setup, pattern);  // calculate feature
            
            for (int j = 0; j<kernelNum; j++) {
                feat[j + cha * kernelNum] = setup->patternProjections[j];
            }
            
            // release memory
            free(setup->patternProjections);
            destroyMatrix(setup->patternImage);
            setup->patternProjections = NULL;
            setup->patternImage = NULL;
        }
        feat /= feat.norm();
        features.push_back(feat);
    }
    
    destroyImageHead(pattern);
    
    assert(features.size() == pts.size());
    return true;
}