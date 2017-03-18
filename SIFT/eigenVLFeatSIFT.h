//
//  eigenVLFeatSIFT.h
//  RGBD_RF
//
//  Created by jimmy on 2016-08-21.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __RGBD_RF__eigenVLFeatSIFT__
#define __RGBD_RF__eigenVLFeatSIFT__

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include "vl_sift_param.h"
#include <vector>
#include <Eigen/Dense>
#include <memory>

using std::vector;

class sift_keypoint
{
public:
    //: Constructor
    sift_keypoint(){;}
    
    //: Destructor
    ~sift_keypoint(){;}
    
    //: Accessor for the x location
    double location_x() const {return location_x_; }
    //: Accessor for the y location
    double location_y() const {return location_y_; }
    
    //: Accessor for the scale
    double scale() const {return scale_; }
    //: Accessor for the orientation
    double orientation() const {return orientation_; }
    
    //: Mutator for the x location
    void set_location_x(double x) { location_x_ = x; }
    //: Mutator for the y location
    void set_location_y(double y) { location_y_ = y; }
    //: Mutator for the scale
    void set_scale(double s) { scale_ = s; }
    //: Mutator for the orientation
    void set_orientation(double o) { orientation_ = o; }
    
    void set_descriptor(const float * data, const int dim)
    {
        descriptor_ = Eigen::VectorXf(dim);
        for (int i = 0; i<dim; i++) {
            descriptor_[i] = data[i];
        }
    }
    
    Eigen::VectorXf descriptor() {return descriptor_;}
    
private:
    //: 128 or 64 -dimensional descriptor vector
    Eigen::VectorXf descriptor_;
    
    //: keypoint parameters
    double location_x_;
    double location_y_;
    double scale_;
    double orientation_;
};

class EigenVLFeatSIFT
{
public:
    // extract SIFT with method from vl_feat, to the format of bapl_keypoint_sptr
    static bool extractSIFTKeypoint(const cv::Mat & image,
                                    const vl_feat_sift_parameter &parameter,
                                    vector<std::shared_ptr<sift_keypoint> > & keypoints,
                                    bool verbose = true);
    
    // extract SIFT at locations    
    // isFound: SIFT may not calculate at some location (e.g. near the image border)
    static bool extractKeypointAtLocations(const cv::Mat & image,
                                             const vl_feat_sift_parameter &parameter,
                                             const vector<cv::Point2d> & locatioins,
                                             vector<std::shared_ptr<sift_keypoint> > & keypoints,
                                             bool verbose = true);
    
    // return: each row is a feature, CV_F32C1
    static cv::Mat descriptorToMat(const vector<std::shared_ptr<sift_keypoint> > & keypoints);
    
    static void descriptorToMatrix(const vector<std::shared_ptr<sift_keypoint> > & keypoints,
                                   Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & matrix);
    
    // feature descriptor + image location
    static void descriptorLocationToMatrix(const vector<std::shared_ptr<sift_keypoint> > & keypoints,
                                           Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & matrix);
};


#endif /* defined(__RGBD_RF__eigenVLFeatSIFT__) */
