//
//  cvx_pose_estimation_uncertainty.cpp
//  PointLineReloc
//
//  Created by jimmy on 2017-05-01.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "cvx_pose_estimation_uncertainty.h"
#include "cvxCalib3d.hpp"
#include "vnl_algo.h"
#include <iostream>
#include <Eigen/QR>

using std::cout;
using std::endl;


struct PoseHypotheseLoss
{
    double energy_;              // loss, the small the better
    Eigen::Affine3d affine_;     //              for 3D --> 3D camera to world transformation
    vector<unsigned int> inlier_indices_;         // camera coordinate index
    vector<unsigned int> inlier_candidate_world_pts_indices_; // candidate world point index
    
    PoseHypotheseLoss()
    {
        energy_ = 0.0;
    }
    PoseHypotheseLoss(const double loss)
    {
        energy_  = loss;
    }
    
    PoseHypotheseLoss(const PoseHypotheseLoss & other)
    {
        energy_ = other.energy_;
        affine_ = other.affine_;
        inlier_indices_ = other.inlier_indices_;
        inlier_candidate_world_pts_indices_ = other.inlier_candidate_world_pts_indices_;
    }
    
    bool operator < (const PoseHypotheseLoss & other) const
    {
        return energy_ < other.energy_;
    }
    
    PoseHypotheseLoss & operator = (const PoseHypotheseLoss & other)
    {
        if (&other == this) {
            return *this;
        }
        energy_ = other.energy_;
        affine_ = other.affine_;
        inlier_indices_ = other.inlier_indices_;
        inlier_candidate_world_pts_indices_ = other.inlier_candidate_world_pts_indices_;
        return *this;
    }
};


template <typename MatrixType>
inline typename MatrixType::Scalar logdet(const MatrixType& M, bool use_cholesky = false) {
    using namespace Eigen;
    using std::log;
    typedef typename MatrixType::Scalar Scalar;
    Scalar ld = 0;
    if (use_cholesky) {
        LLT<Matrix<Scalar,Dynamic,Dynamic>> chol(M);
        auto& U = chol.matrixL();
        for (unsigned i = 0; i < M.rows(); ++i)
            ld += log(U(i,i));
        ld *= 2;
    } else {
        PartialPivLU<Matrix<Scalar,Dynamic,Dynamic>> lu(M);
        auto& LU = lu.matrixLU();
        Scalar c = lu.permutationP().determinant(); // -1 or 1
        for (unsigned i = 0; i < LU.rows(); ++i) {
            const auto& lii = LU(i,i);
            if (lii < Scalar(0)) c *= -1;
            ld += log(fabs(lii));
        }
        ld += log(c);
    }
    return ld;
}

// log(p) where p is the probability
static double gaussianLogLikelihood(const Eigen::Vector3d& mean,
                                    const Eigen::Matrix3d& cov,
                                    const Eigen::Matrix3d& precision,
                                    const Eigen::Vector3d& p)
{
    /*
    cout<<"mean "<<mean.transpose()<<endl;
    cout<<"sample point"<<p.transpose()<<endl;
    cout<<"covariance "<<cov<<endl;
    cout<<"precision "<<precision<<endl<<endl;
     */
    
    double likelihood = 0;
    likelihood += -0.5 * 3.0 * log(2.0 * M_PI);
    double det = cov.determinant();
    //likelihood += -0.5 * logdet(cov, true);
    likelihood += -0.5 * log(det);
    Eigen::Vector3d dif = p - mean;
    double maha_dist = dif.transpose() * precision * dif;
    likelihood += -0.5 * maha_dist;
    return likelihood;
}

bool CvxPoseEstimationUncertainty::preemptiveRANSAC3DOneToMany(const vector<Eigen::Vector3d>& input_camera_pts,
                                                               const vector<vector<Eigen::Vector3d> >& input_world_pts,
                                                               const vector<vector<Eigen::Matrix3d> >& input_world_pts_covariance,
                                                               const PreemptiveRANSACUncParameter& param,
                                                               Eigen::Affine3d& output_camera_pose)
{
    // 1. initialize the camera pose
    assert(input_camera_pts.size() == input_world_pts.size());
    assert(input_camera_pts.size() == input_world_pts_covariance.size());
    
    // some magic numbers
    if (input_camera_pts.size() < param.sample_num_) {
        return false;
    }
    
    const int num_iteration = 256;
    const int K = 128;
    const int N = (int)input_camera_pts.size();
    const int B = param.sample_num_;
    
    // covariance to precision matrix
    vector<vector<Eigen::Matrix3d> > input_world_pts_precision(N); // todo
    vector<bool> valid_indices(N, true);
    for (unsigned i = 0; i < input_world_pts_covariance.size(); i++) {
        for (int j = 0; j<input_world_pts_covariance[i].size(); j++) {
            Eigen::ColPivHouseholderQR<Eigen::Matrix3d> qr(input_world_pts_covariance[i][j]);
            if (qr.isInvertible() && input_world_pts_covariance[i][j].determinant() > 0.0) {  // positive definitive
                Eigen::Matrix3d precision = input_world_pts_covariance[i][j].inverse();
                input_world_pts_precision[i].push_back(precision);
            }
            else {
                //cout<<"uninvertable matrix:\n "<<input_world_pts_covariance[i][j]<<endl;
                valid_indices[i] = false;  // mark the invalid points index
            }
        }
    }
    
    //
    int invalid_num = 0;
    for (int i = 0; i<valid_indices.size(); i++) {
        if (!valid_indices[i]) {
            invalid_num++;
        }
    }
    printf("invalid world point covariance percentage is %lf\n", 1.0 * invalid_num/valid_indices.size());
    
    vector<Eigen::Affine3d > affine_candidate;
    for (int i = 0; i<num_iteration; i++) {
        
        int k1 = 0;
        int k2 = 0;
        int k3 = 0;
        int k4 = 0;
        
        do{
            k1 = rand()%N;
            k2 = rand()%N;
            k3 = rand()%N;
            k4 = rand()%N;
        }while (k1 == k2 || k1 == k3 || k1 == k4 ||
                k2 == k3 || k2 == k4 || k3 == k4);
        
        vector<Eigen::Vector3d> sampled_camera_pts;
        vector<Eigen::Vector3d> sampled_wld_pts;
        
        sampled_camera_pts.push_back(input_camera_pts[k1]);
        sampled_camera_pts.push_back(input_camera_pts[k2]);
        sampled_camera_pts.push_back(input_camera_pts[k3]);
        sampled_camera_pts.push_back(input_camera_pts[k4]);
        
        sampled_wld_pts.push_back(input_world_pts[k1][0]);
        sampled_wld_pts.push_back(input_world_pts[k2][0]);
        sampled_wld_pts.push_back(input_world_pts[k3][0]);
        sampled_wld_pts.push_back(input_world_pts[k4][0]);
        
        Eigen::Affine3d affine = CvxCalib3D::KabschTransform(sampled_camera_pts, sampled_wld_pts);
        affine_candidate.push_back(affine);
        if (affine_candidate.size() > K) {
            //printf("initialization repeat %d times\n", i);
            break;
        }
    }
    //printf("init camera parameter number is %lu\n", affine_candidate.size());
    
    // 2. optimize camera pose by minimize Mahalanobis distance
    vector<PoseHypotheseLoss> hypotheses;
    for (int i = 0; i<affine_candidate.size(); i++) {
        PoseHypotheseLoss hyp(0.0);
        hyp.affine_ = affine_candidate[i];
        hypotheses.push_back(hyp);
    }
    
    const double threshold = param.dist_threshold_;
    while (hypotheses.size() > 1) {
        // sample random set
        vector<int> sampled_indices;
        for (int i = 0; i<B; i++) {
            int index = rand()%N;
            if (valid_indices[index]) {
                sampled_indices.push_back(index);
            }
        }
        
        // pick inliers in each hypothese
        for (int i = 0; i<hypotheses.size(); i++) {
            // evaluate the accuracy by checking transformation
            Eigen::Affine3d cur_affine = hypotheses[i].affine_;
           
            for (int j = 0; j<sampled_indices.size(); j++) {
                int index = sampled_indices[j];
                Eigen::Vector3d p = input_camera_pts[index];
                Eigen::Vector3d transformed_pt = cur_affine * p;  // transform from camera coordiante to world coordinate
                
                double min_dis = threshold * 2;
                int min_index = -1;
                // loop each candidate world points
                for (int k = 0; k<input_world_pts[index].size(); k++) {
                    Eigen::Vector3d wld_p = input_world_pts[index][k];
                    double dist = (wld_p - transformed_pt).norm();
                    if (dist < min_dis) {
                        min_dis = dist;
                        min_index = k;
                    }
                    //cout<<"distance is %lf: "<<dist<<endl;
                }// k                
               
                // update inlier index
                if (min_dis < threshold) {
                    hypotheses[i].inlier_indices_.push_back(index);
                    hypotheses[i].inlier_candidate_world_pts_indices_.push_back(min_index);
                }
                else {
                    hypotheses[i].energy_ += 1.0;
                }
            }// j
            assert(hypotheses[i].inlier_indices_.size() == hypotheses[i].inlier_candidate_world_pts_indices_.size());
        }// i
        
        std::sort(hypotheses.begin(), hypotheses.end());
        hypotheses.resize(hypotheses.size()/2);
        
        for (int i = 0; i<hypotheses.size(); i++) {
          //  printf("after: loss is %lf\n", hypotheses[i].energy_);
          //  printf("inlier number is %lu\n", hypotheses[i].inlier_indices_.size());
        }
        
        // refine camera pose by inliers
        // refine by inliers
        for (int i = 0; i<hypotheses.size(); i++) {
            // number of inliers is larger than minimum configure
            if (hypotheses[i].inlier_indices_.size() > 7) {
                vector<Eigen::Vector3d> inlier_camera_pts;
                vector<Eigen::Vector3d> inlier_world_pts;
                vector<Eigen::Matrix3d> inlier_wordl_pts_precision;
                
                for (int j = 0; j<hypotheses[i].inlier_indices_.size(); j++) {
                    int index = hypotheses[i].inlier_indices_[j];
                    int wld_sub_index = hypotheses[i].inlier_candidate_world_pts_indices_[j];
                    inlier_camera_pts.push_back(input_camera_pts[index]);
                    inlier_world_pts.push_back(input_world_pts[index][wld_sub_index]);
                    inlier_wordl_pts_precision.push_back(input_world_pts_precision[index][wld_sub_index]);
                }
                //printf("inlier number %lu\n", inlier_camera_pts.size());
                
                Eigen::Affine3d refined_pose;                
                bool is_estimated = VnlAlgo::estimateCameraPoseWithUncertainty(inlier_camera_pts,
                                                                               inlier_world_pts,
                                                                               inlier_wordl_pts_precision,
                                                                               hypotheses[i].affine_,
                                                                               refined_pose);
                if (is_estimated) {
                    hypotheses[i].affine_ = refined_pose;
                }
                hypotheses[i].inlier_indices_.clear();
                hypotheses[i].inlier_candidate_world_pts_indices_.clear();
            }
        }// i
    } // while
    assert(hypotheses.size() == 1);
    
    output_camera_pose = hypotheses.front().affine_;    
    
    return true;
}