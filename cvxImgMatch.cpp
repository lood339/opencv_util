//
//  cvxImgMatch.cpp
//  RGBD_RF
//
//  Created by jimmy on 2016-09-06.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "cvxImgMatch.h"
#include "eigenVLFeatSIFT.h"
#include "eigenFlann.h"

void CvxImgMatch::SIFTMatching(const cv::Mat & srcImg, const cv::Mat & dstImg,
                               const SIFTMatchingParameter & param,
                               vector<cv::Point2d> & srcPts, vector<cv::Point2d> & dstPts)
{
    const double ratio_threshold = 0.7;
    double feature_distance_threshold = 0.5;
    
    vl_feat_sift_parameter sift_param;
    sift_param.edge_thresh = 10;
    sift_param.dim = 128;
    sift_param.nlevels = 3;
    
    vector<std::shared_ptr<sift_keypoint> > src_keypoints;
    vector<std::shared_ptr<sift_keypoint> > dst_keypoints;
    EigenVLFeatSIFT::extractSIFTKeypoint(srcImg, sift_param, src_keypoints, false);
    EigenVLFeatSIFT::extractSIFTKeypoint(dstImg, sift_param, dst_keypoints, false);
    
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> src_descriptors;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dst_descriptors;
    
    
    EigenVLFeatSIFT::descriptorToMatrix(src_keypoints, src_descriptors);
    EigenVLFeatSIFT::descriptorToMatrix(dst_keypoints, dst_descriptors);
    
    EigenFlann32F flann32;
    flann32.setData(dst_descriptors);
    
    vector<vector<int> >  indices;       // index of src descriptors
    vector<vector<float> > dists;
    flann32.search(src_descriptors, indices, dists, 2);
    
    for (int i = 0; i<src_keypoints.size(); i++) {
        double dis1 = dists[i][0];
        double dis2 = dists[i][1];        
        if (dis1 < feature_distance_threshold && dis1 < dis2 * ratio_threshold) {
            int dst_index = indices[i][0];
            cv::Point2d src_pt(src_keypoints[i]->location_x(), src_keypoints[i]->location_y());
            cv::Point2d dst_pt(dst_keypoints[dst_index]->location_x(), dst_keypoints[dst_index]->location_y());
            srcPts.push_back(src_pt);
            dstPts.push_back(dst_pt);
        }
    }
    assert(srcPts.size() == dstPts.size());
}

void CvxImgMatch::ORBMatching(const cv::Mat & srcImg, const cv::Mat & dstImg,
                              vector<cv::Point2d> & srcPts, vector<cv::Point2d> & dstPts)
{
    
}

