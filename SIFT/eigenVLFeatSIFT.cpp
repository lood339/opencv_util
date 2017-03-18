//
//  eigenVLFeatSIFT.cpp
//  RGBD_RF
//
//  Created by jimmy on 2016-08-21.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "eigenVLFeatSIFT.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <vl/sift.h>
#include "sift_64.h"
#include <vl/imopv.h>
#include <iostream>
//#include "vl_detect.h"

using std::cout;
using std::endl;

bool EigenVLFeatSIFT::extractSIFTKeypoint(const cv::Mat & image,
                                          const vl_feat_sift_parameter &param,
                                          vector<std::shared_ptr<sift_keypoint>> & keypoints,
                                          bool verbose)
{
    assert(image.type() == CV_8UC3 || image.type() == CV_8UC1);
    
    cv::Mat grey;
    if (image.channels() == 1) {
        grey = image;
    }
    else {
        cv::cvtColor(image, grey, CV_BGR2GRAY);
    }
    
    const int width  = grey.cols;
    const int height = grey.rows;
    const int noctaves = param.noctaves; // maximum octatve possible
    const int nlevels = param.nlevels;
    const int o_min = 0;   //first octave index
    const int dim = param.dim;
    assert(dim == 128);
    
    // create a filter to process the image
    VlSiftFilt *filt = NULL;
    if (dim == 128) {
        filt = vl_sift_new (width, height, noctaves, nlevels, o_min) ;
    }
    else {
        filt = vl_sift_new_64(width, height, noctaves, nlevels, o_min);
    }
    
    
    double   edge_thresh  = param.edge_thresh;
    double   peak_thresh  = param.peak_thresh;
    double   magnif       = param.magnif ;
    double   norm_thresh  = param.norm_thresh;
    double   window_size  = param.window_size;
    
    if (peak_thresh >= 0) vl_sift_set_peak_thresh (filt, peak_thresh) ;
    if (edge_thresh >= 0) vl_sift_set_edge_thresh (filt, edge_thresh) ;
    if (norm_thresh >= 0) vl_sift_set_norm_thresh (filt, norm_thresh) ;
    if (magnif      >= 0) vl_sift_set_magnif      (filt, magnif) ;
    if (window_size >= 0) vl_sift_set_window_size (filt, window_size) ;
    
    // data from image
    vl_sift_pix *fdata = (vl_sift_pix *)malloc(width * height * sizeof (vl_sift_pix));
    for (int y = 0; y<height; y++) {
        for (int x = 0; x<width; x++) {
            int idx = y * width + x;
            fdata[idx] = grey.at<unsigned char>(y, x);
        }
    }
    
    //                                             Process each octave
    bool isFirst = true ;
    vl_bool err = VL_ERR_OK;
    
    int nangles = 0;
    double angles[4] = {0.0f};
    float descriptor[128] = {0.0f};
    while (1) {
        if (isFirst) {
            isFirst = false;
            err = vl_sift_process_first_octave (filt, fdata) ;
        } else {
            err = vl_sift_process_next_octave  (filt) ;
        }
        if(err == VL_ERR_EOF)
        {
            break;
        }
        
      //  printf("filter sigma0 sigman: %f %f, octave width %d \n", filt->sigma0, filt->sigman, filt->octave_width);
        
        vl_sift_detect (filt);
        
        VlSiftKeypoint const * keys  = vl_sift_get_keypoints(filt) ;
        int nkeys = vl_sift_get_nkeypoints (filt) ;
        
        bool is_first = false;
        for (int i = 0; i<nkeys; i++) {
            VlSiftKeypoint const * curKey = keys + i;
            
            // Depending on the symmetry of the keypoint appearance, determining the orientation can be ambiguous. SIFT detectors have up to four possible orientations
            // vl_sift_calc_keypoint_orientation
            nangles = vl_sift_calc_keypoint_orientations(filt, angles, curKey) ;
            assert(curKey->o == filt->o_cur);
            
            for (int q = 0 ; q < nangles ; q++) {
                if (dim == 128) {
                    vl_sift_calc_keypoint_descriptor(filt, &descriptor[0], curKey, angles[q]);
                }
                else {
                    vl_sift_calc_keypoint_descriptor_64(filt, &descriptor[0], curKey, angles[q]);
                }
                
                std::shared_ptr<sift_keypoint> pKeypoint(new sift_keypoint);
                
                double x = curKey->x;
                double y = curKey->y;
                double s = curKey->sigma;
                double o = angles[q];
                
                pKeypoint->set_location_x(x);
                pKeypoint->set_location_y(y);
                pKeypoint->set_scale(s);
                pKeypoint->set_orientation(o);
                pKeypoint->set_descriptor(&descriptor[0], dim);
                keypoints.push_back(pKeypoint);
                
                if (!is_first) {
                 //   printf("key point scale %f\n", curKey->sigma);
                    is_first = true;
                }
            }
        }
    }
    
    vl_sift_delete(filt);
    free(fdata);
    
    if(verbose){
        cout<<"Found "<<keypoints.size()<<" keypoints."<<endl;
    }
    return true;
}


bool EigenVLFeatSIFT::extractKeypointAtLocations(const cv::Mat & image,
                                                 const vl_feat_sift_parameter &param,
                                                 const vector<cv::Point2d> & locatioins,                                                 
                                                 vector<std::shared_ptr<sift_keypoint> > & keypoints,
                                                 bool verbose)
{
    assert(locatioins.size() > 0);
    
    assert(image.type() == CV_8UC3 || image.type() == CV_8UC1);
    
    cv::Mat grey;
    if (image.channels() == 1) {
        grey = image;
    }
    else {
        cv::cvtColor(image, grey, CV_BGR2GRAY);
    }
    
    const int width  = grey.cols;
    const int height = grey.rows;
    const int noctaves = -1; // maximum octatve possible
    const int nlevels = 3;
    const int o_min = 0;   //first octave index
    const int dim = param.dim;
    assert(dim == 64);
    
    // create a filter to process the image
    VlSiftFilt *filt = vl_sift_new_64 (width, height, noctaves, nlevels, o_min) ;
    double   edge_thresh  = param.edge_thresh;
    double   peak_thresh  = param.peak_thresh;
    double   magnif       = param.magnif ;
    double   norm_thresh  = param.norm_thresh;
    double   window_size  = param.window_size;
    
    if (peak_thresh >= 0) vl_sift_set_peak_thresh (filt, peak_thresh) ;
    if (edge_thresh >= 0) vl_sift_set_edge_thresh (filt, edge_thresh) ;
    if (norm_thresh >= 0) vl_sift_set_norm_thresh (filt, norm_thresh) ;
    if (magnif      >= 0) vl_sift_set_magnif      (filt, magnif) ;
    if (window_size >= 0) vl_sift_set_window_size (filt, window_size) ;
    
    // data from image
    vl_sift_pix *fdata = (vl_sift_pix *)malloc(width * height * sizeof (vl_sift_pix));
    for (int y = 0; y<height; y++) {
        for (int x = 0; x<width; x++) {
            int idx = y * width + x;
            fdata[idx] = grey.at<unsigned char>(y, x);
        }
    }
    
    // smoothed image data
    float *smoothed_data = (float *)malloc(width * height * sizeof(float));
    memset(smoothed_data, 0, width * height * sizeof(smoothed_data[0]));
    vl_imsmooth_f(smoothed_data, width, fdata, width, height, width, filt->sigma0, filt->sigma0);
    
    // magnitude, angle, magnitude, angle..
    float * gradientAmplitudeAngle = (float *)malloc(2 * sizeof(float) * width * height);
    memset(gradientAmplitudeAngle, 0, 2 * sizeof(float) * width * height);
    vl_imgradient_polar_f(gradientAmplitudeAngle, gradientAmplitudeAngle + 1, 2, 2 * width, smoothed_data, width, height, width);
    
    
    for (int i = 0; i<locatioins.size(); i++) {
        double x = locatioins[i].x;
        double y = locatioins[i].y;
        double sigma = filt->sigma0; // ?
        double angle = 0.0;
        vl_sift_pix descr[64] = {0.0f};
        
        vl_sift_calc_raw_descriptor_64(filt, gradientAmplitudeAngle, descr, width, height, x, y, sigma, angle);
        
        std::shared_ptr<sift_keypoint> pKeypoint(new sift_keypoint);
        pKeypoint->set_location_x(x);
        pKeypoint->set_location_y(y);
        pKeypoint->set_scale(sigma);
        pKeypoint->set_orientation(angle);
        pKeypoint->set_descriptor(descr, dim);
        
        keypoints.push_back(pKeypoint);
    }     
    
    vl_sift_delete(filt);
    free(fdata);
    free(gradientAmplitudeAngle);
    free(smoothed_data);
    
    if(verbose){
        cout<<"Found "<<keypoints.size()<<" keypoints."<<endl;
    }
    assert(keypoints.size() == locatioins.size());
    return true;
}

cv::Mat EigenVLFeatSIFT::descriptorToMat(const vector<std::shared_ptr<sift_keypoint> > & keypoints)
{
    cv::Mat features = cv::Mat((int)keypoints.size(), (int)keypoints[0]->descriptor().size(), CV_32FC1);
    
    for (int i = 0; i<keypoints.size(); i++) {
        Eigen::VectorXf descriptor = keypoints[i]->descriptor();
        for (int j = 0; j<descriptor.size(); j++) {
            features.at<float>(i, j) = descriptor[j];
        }
    }
    return features;
}

void EigenVLFeatSIFT::descriptorToMatrix(const vector<std::shared_ptr<sift_keypoint> > & keypoints,
                                         Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & matrix)
{
    assert(keypoints.size() > 0);
    const int rows = (int)keypoints.size();
    const int cols = (int)(keypoints[0]->descriptor().size());
    matrix.resize(rows, cols);
    
    for (int i = 0; i<keypoints.size(); i++) {
        Eigen::VectorXf feat = keypoints[i]->descriptor();
        for (int j = 0; j<feat.size(); j++) {
            matrix(i, j) = feat[j];
        }
    }
}

void EigenVLFeatSIFT::descriptorLocationToMatrix(const vector<std::shared_ptr<sift_keypoint> > & keypoints,
                                                 Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & matrix)
{
    assert(keypoints.size() > 0);
    const int rows = (int)keypoints.size();
    const int cols = (int)(keypoints[0]->descriptor().size()) + 2;
    matrix.resize(rows, cols);
    
    for (int i = 0; i<keypoints.size(); i++) {
        Eigen::VectorXf feat = keypoints[i]->descriptor();
        for (int j = 0; j<feat.size(); j++) {
            matrix(i, j) = feat[j];
        }
        matrix(i, feat.size()) = keypoints[i]->location_x();
        matrix(i, feat.size() + 1) = keypoints[i]->location_y();
    }
}




