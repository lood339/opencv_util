//
//  cvx_kvld.cpp
//  MAVGoogleImageMatching
//
//  Created by jimmy on 2015-11-28.
//  Copyright © 2015 jimmy. All rights reserved.
//

#include "cvx_kvld.hpp"
#include "vxlOpenCV.h"
#include "kvld.h"
#include <bapl/bapl_keypoint_sptr.h>
#include <bapl/bapl_lowe_keypoint.h>


/*
static int Convert_image(const cv::Mat& In, cv::Mat& imag)//convert only gray scale image of opencv
{
    imag = cv::Mat(In.rows, In.cols, CV_32FC1);
    int cn = In.channels();
    if (cn == 1)//gray scale
    {
        for (int i = 0; i < In.rows; ++i)
        {
            for (int j = 0; j < In.cols; ++j)
            {
                imag.at<float>(i, j) = In.at<unsigned char>(i, j);
            }
        }
    }
    else
    {
        for (int i = 0; i < In.rows; ++i)
        {
            for (int j = 0; j < In.cols; ++j)
            {
                //imag.at<float>(i, j) = (float(pixelPtr[(i*In.cols + j)*cn + 0]) * 29 + float(pixelPtr[(i*In.cols + j)*cn + 1]) * 150 + float(pixelPtr[(i*In.cols + j)*cn + 2]) * 77) / 255;
                //why not using uniform weights of the channels?
            }
        }
    }
    return 0;
}
 */

static int convert_image(const cv::Mat & in, cv::Mat & out)
{
    out = cv::Mat(in.rows, in.cols, CV_32FC1);
    cv::Mat gray;
    if (in.channels() == 3) {
        cv::cvtColor(in, gray, CV_BGR2GRAY);
    }
    else if(in.channels() == 1)
    {
        gray = in;
    }
    else
    {
        assert(0);
    }
    assert(gray.channels() == 1);
    
    for (int i = 0; i < gray.rows; ++i)
    {
        for (int j = 0; j < gray.cols; ++j)
        {
            out.at<float>(i, j) = gray.at<unsigned char>(i, j);
        }
    }
    return 0;
}

bool cvx_kvld::kvld_matching(const vil_image_view<vxl_byte> & image1,
                             const vil_image_view<vxl_byte> & image2,
                             vcl_vector<bapl_key_match> & final_matches,
                             vcl_vector<bool> & is_valid,
                             const cvx_kvld_parameter & param)
{
    // feature point set 1
    std::vector<VLDKeyPoint> F1;
    for (int i = 0; i<param.keypoint_1.size(); i++) {
        VLDKeyPoint vld_pt;
        bapl_lowe_keypoint_sptr sift = dynamic_cast<bapl_lowe_keypoint *>(param.keypoint_1[i].as_pointer());
        assert(sift);
        vld_pt.cvKeyPoint.size = sift->scale();
        vld_pt.cvKeyPoint.angle = sift->orientation();
        vld_pt.cvKeyPoint.pt = cv::Point2f(sift->location_i(), sift->location_j());
        F1.push_back(vld_pt);
    }
    
    // feature point set 2
    std::vector<VLDKeyPoint> F2;
    for (int i = 0; i<param.keypoint_2.size(); i++) {
        VLDKeyPoint vld_pt;
        bapl_lowe_keypoint_sptr sift = dynamic_cast<bapl_lowe_keypoint *>(param.keypoint_2[i].as_pointer());
        assert(sift);
        vld_pt.cvKeyPoint.size = sift->scale();
        vld_pt.cvKeyPoint.angle = sift->orientation();
        vld_pt.cvKeyPoint.pt = cv::Point2f(sift->location_i(), sift->location_j());
        F2.push_back(vld_pt);
    }
    
    std::cout << "sift:: 1st image: " << F1.size() << " keypoints" << std::endl;
    std::cout << "sift:: 2nd image: " << F2.size() << " keypoints" << std::endl;
    
    
    std::vector<cv::DMatch> matches;
    std::vector<cv::DMatch> matchesFiltered;
    std::vector<double> vec_score;
    
    for (int i = 0; i<param.matches.size(); i++) {
        matches.push_back(cv::DMatch(param.matches[i].first, param.matches[i].second, 0.0));
    }
    
    std::cout << "K-VLD starts with " << matches.size() << " matches" << std::endl;
    
    //In order to illustrate the gvld(or vld)-consistant neighbors, the following two parameters has been externalized as inputs of the function KVLD.
    Matrixf E = Matrixf::ones(matches.size(), matches.size(), CV_32FC1)*(-1);
    // gvld-consistency matrix, intitialized to -1,  >0 consistency value, -1=unknow, -2=false
    
    is_valid = std::vector<bool>(matches.size(), true);// indices of match in the initial matches, if true at the end of KVLD, a match is kept.
    size_t it_num = 0;
    KvldParameters kvldparameters;//initial parameters of KVLD
    
    cv::Mat cv_image1 = VxlOpenCVImage::cv_image(image1);
    cv::Mat cv_image2 = VxlOpenCVImage::cv_image(image2);
    
   // int convert_image(const cv::Mat & in, cv::Mat & out)
    cv::Mat cv_float_image1;
    cv::Mat cv_float_image2;
    convert_image(cv_image1, cv_float_image1);
    convert_image(cv_image2, cv_float_image2);
    
    while (it_num < 5 && kvldparameters.inlierRate>KVLD(cv_float_image1, cv_float_image2, F1, F2,
                                                        matches, matchesFiltered, vec_score, E, is_valid, kvldparameters))
    {
        kvldparameters.inlierRate /= 2;
        kvldparameters.rang_ratio = sqrt(2.0f);
        std::cout << "low inlier rate, re-select matches with new rate=" << kvldparameters.inlierRate << std::endl;
        if (matchesFiltered.size() == 0) kvldparameters.K = 2;
        it_num++;
    }
    std::cout << "K-VLD filter ends with " << matchesFiltered.size() << " selected matches" << std::endl;
    
    final_matches.clear();
    for (int i = 0; i<is_valid.size(); i++) {
        if (is_valid[i]) {
            int idx1 = param.matches[i].first;
            int idx2 = param.matches[i].second;
            bapl_key_match one_match(param.keypoint_1[idx1], param.keypoint_2[idx2]);
            final_matches.push_back(one_match);
        }
    }
    
    
    /*
    cv::Mat image1color, image2color, concat;//for visualization
    image1color = cv::imread(mavImgFilePath, CV_LOAD_IMAGE_COLOR);
    image2color = cv::imread(streetImgFilePath, CV_LOAD_IMAGE_COLOR);
    
    //=============== Read SIFT points =================//
    std::cout << "Loading SIFT features" << std::endl;
    
    read_detectors(datasetPath + "mav_keypoint/keypoint_" + mavImgNumStr + ".txt", F1);
    read_detectors(datasetPath + "street_keypoint/keypoint_" + streetImgNumStr + ".txt", F2);
    
    std::cout << "sift:: 1st image: " << F1.size() << " keypoints" << std::endl;
    std::cout << "sift:: 2nd image: " << F2.size() << " keypoints" << std::endl;
    
    //=============== load initial matching ====================//
    std::vector<cv::DMatch> matches;
    read_matches(datasetPath + "initial_matchings/" + fileName, matches);
    
    //===============================  KVLD method ==================================//
    std::cout << "K-VLD starts with " << matches.size() << " matches" << std::endl;
    
    std::vector<cv::DMatch> matchesFiltered;
    std::vector<double> vec_score;
    
    //In order to illustrate the gvld(or vld)-consistant neighbors, the following two parameters has been externalized as inputs of the function KVLD.
    Matrixf E = Matrixf::ones(matches.size(), matches.size(), CV_32FC1)*(-1);
    // gvld-consistency matrix, intitialized to -1,  >0 consistency value, -1=unknow, -2=false
    
    std::vector<bool> valide(matches.size(), true);// indices of match in the initial matches, if true at the end of KVLD, a match is kept.
    
    size_t it_num = 0;
    KvldParameters kvldparameters;//initial parameters of KVLD
    
    while (it_num < 5 && kvldparameters.inlierRate>KVLD(image1, image2, F1, F2,
                                                        matches, matchesFiltered, vec_score, E, valide, kvldparameters))
    {
        kvldparameters.inlierRate /= 2;
        kvldparameters.rang_ratio = sqrt(2.0f);
        std::cout << "low inlier rate, re-select matches with new rate=" << kvldparameters.inlierRate << std::endl;
        if (matchesFiltered.size() == 0) kvldparameters.K = 2;
        it_num++;
    }
    std::cout << "K-VLD filter ends with " << matchesFiltered.size() << " selected matches" << std::endl;
    
    return true;
     */
    
    return true;
}