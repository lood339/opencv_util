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
    vector<unsigned int> inlier_indices_;                     // camera coordinate point index
    vector<unsigned int> inlier_candidate_world_pts_indices_; // candidate world point index
    vector<unsigned int> line_inlier_indices_;                // line in camera coordinate
    vector<vector<unsigned int> > point_on_line_inlier_index_groups_;  // line points in world coordinate
    
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
        line_inlier_indices_ = other.line_inlier_indices_;
        point_on_line_inlier_index_groups_ = other.point_on_line_inlier_index_groups_;
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
        line_inlier_indices_ = other.line_inlier_indices_;
        point_on_line_inlier_index_groups_ = other.point_on_line_inlier_index_groups_;
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
    
    vector<Eigen::Affine3d > affine_candidate = CvxPoseEstimationUncertainty::sampleCameraPose(input_camera_pts,
                                                                                               input_world_pts, num_iteration, K);
    
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

vector<Eigen::Affine3d>
CvxPoseEstimationUncertainty::sampleCameraPose(const vector<Eigen::Vector3d>& camera_pts,
                                                                       const vector<vector<Eigen::Vector3d> >& world_pts,
                                                                       const int num_iteration, const int num_poses)
{
    assert(num_iteration >= num_poses);
    
    const int N = (int)camera_pts.size();
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
        
        sampled_camera_pts.push_back(camera_pts[k1]);
        sampled_camera_pts.push_back(camera_pts[k2]);
        sampled_camera_pts.push_back(camera_pts[k3]);
        sampled_camera_pts.push_back(camera_pts[k4]);
        
        sampled_wld_pts.push_back(world_pts[k1][0]);
        sampled_wld_pts.push_back(world_pts[k2][0]);
        sampled_wld_pts.push_back(world_pts[k3][0]);
        sampled_wld_pts.push_back(world_pts[k4][0]);
        
        Eigen::Affine3d affine = CvxCalib3D::KabschTransform(sampled_camera_pts, sampled_wld_pts);
        affine_candidate.push_back(affine);
        if (affine_candidate.size() > num_poses) {
            //printf("initialization repeat %d times\n", i);
            break;
        }
    }
    return affine_candidate;    
}

bool CvxPoseEstimationUncertainty::preemptiveRANSACPointAndLine(const vector<Eigen::Vector3d>& input_camera_pts,
                                                                const vector<vector<Eigen::Vector3d> >& input_world_pts,
                                                                const vector<vector<Eigen::Matrix3d> >& input_world_pts_covariance,
                                  
                                                                const vector<vector< Eigen::Vector3d> > & input_world_line_pts_group,
                                                                const vector<vector< Eigen::Matrix3d> > & input_world_line_pts_covariance,
                                                                const vector<std::pair<Eigen::Vector3d, Eigen::Vector3d> >& input_camera_lines,
                                                                const PreemptiveRANSACUncParameter & param,
                                                                Eigen::Affine3d& output_camera_pose)
{    
    // 1. initialize the camera pose
    assert(input_camera_pts.size() == input_world_pts.size());
    assert(input_camera_pts.size() == input_world_pts_covariance.size());
    assert(input_world_line_pts_group.size() == input_world_line_pts_covariance.size());
    assert(input_world_line_pts_group.size() == input_camera_lines.size());
    
    // some magic numbers
    if (input_camera_pts.size() < param.sample_num_) {
        return false;
    }
    
    const int num_iteration = 256;
    const int K = 128;
    const int N = (int)input_camera_pts.size();
    const int B = param.sample_num_;
    const int line_num = (int)input_world_line_pts_group.size();
    
    // covariance to precision matrix
    vector<vector<Eigen::Matrix3d> > input_world_pts_precision(N); // todo
    vector<bool> valid_world_point_indices(N, true);
    for (unsigned i = 0; i < input_world_pts_covariance.size(); i++) {
        for (int j = 0; j<input_world_pts_covariance[i].size(); j++) {
            Eigen::ColPivHouseholderQR<Eigen::Matrix3d> qr(input_world_pts_covariance[i][j]);
            if (qr.isInvertible() && input_world_pts_covariance[i][j].determinant() > 0.0) {  // positive definitive
                Eigen::Matrix3d precision = input_world_pts_covariance[i][j].inverse();
                input_world_pts_precision[i].push_back(precision);
            }
            else {
                //cout<<"uninvertable matrix:\n "<<input_world_pts_covariance[i][j]<<endl;
                valid_world_point_indices[i] = false;  // mark the invalid points index
            }
        }
    }
    
    // convariance matrix to precision matrix
    Eigen::Matrix3d min_cov = param.min_variance_ * Eigen::Matrix3d::Identity();
    cout<<"add \n"<<min_cov<<endl;
    cout<<"to points on line covariance matrix for numerical stability"<<endl;
    vector<vector< Eigen::Matrix3d> > input_world_line_pts_precision(input_world_line_pts_group.size());
    vector<vector<bool > > valid_point_on_line(input_world_line_pts_group.size());
    int total_point_on_line_num = 0;
    for (int i = 0; i<input_world_line_pts_covariance.size(); i++) {
        for (int j = 0; j<input_world_line_pts_covariance[i].size(); j++) {
            Eigen::Matrix3d cov = input_world_line_pts_covariance[i][j] + min_cov;
            Eigen::ColPivHouseholderQR<Eigen::Matrix3d> qr(cov);
            if (qr.isInvertible() && cov.determinant() > 0.0) {
                Eigen::Matrix3d precision = cov.inverse();
                input_world_line_pts_precision[i].push_back(precision);
                valid_point_on_line[i].push_back(true);
            }
            else {
                Eigen::Matrix3d precision = Eigen::Matrix3d::Identity(); // dummy data
                input_world_line_pts_precision[i].push_back(precision);
                valid_point_on_line[i].push_back(false);
                
                //cout<<"covariance matrix is: \n"<<input_world_line_pts_covariance[i][j]<<endl<<endl;
            }
            total_point_on_line_num++;
        }
    }
    
    // for computational convenience
    vector<Eigen::ParametrizedLine<double, 3> > input_camera_line_infinite;
    for (int i = 0; i<input_camera_lines.size(); i++) {
        Eigen::ParametrizedLine<double, 3> line(input_camera_lines[i].first, input_camera_lines[i].second);
        input_camera_line_infinite.push_back(line);
    }
    
    // invalid point number
    int invalid_num = 0;
    for (int i = 0; i<valid_world_point_indices.size(); i++) {
        if (!valid_world_point_indices[i]) {
            invalid_num++;
        }
    }
    int invalid_point_on_line_num = 0;
    for (int i = 0; i<valid_point_on_line.size(); i++) {
        for (int j = 0; j<valid_point_on_line[i].size(); j++) {
            if (!valid_point_on_line[i][j]) {
                invalid_point_on_line_num++;
            }
        }
    }
    printf("invalid world point covariance percentage: %lf\n", 1.0 * invalid_num/valid_world_point_indices.size());
    printf("invalid point on line number: %d, percentage: %lf\n", invalid_point_on_line_num, 1.0 * invalid_point_on_line_num/total_point_on_line_num);
    
    vector<Eigen::Affine3d> affine_candidate = CvxPoseEstimationUncertainty::sampleCameraPose(input_camera_pts,
                                                                                              input_world_pts, num_iteration, K);
    
    // 2. optimize camera pose by minimize Mahalanobis distance
    vector<PoseHypotheseLoss> hypotheses;
    for (int i = 0; i<affine_candidate.size(); i++) {
        PoseHypotheseLoss hyp(0.0);
        hyp.affine_ = affine_candidate[i];
        hypotheses.push_back(hyp);
    }
    
    const double threshold = param.dist_threshold_;
    const double point_on_line_wt = param.line_point_wt_;
    const int min_line_support_num = param.min_line_support_num_;
    const int line_sample_num = param.line_sample_num_;
    const int joint_opt_num = param.point_line_joint_opt_num_;
    while (hypotheses.size() > 1) {
        // sample random set of points
        vector<int> sampled_indices;
        for (int i = 0; i<B; i++) {
            int index = rand()%N;
            if (valid_world_point_indices[index]) {
                sampled_indices.push_back(index);
            }
        }
        
        // sample random set of lines
        vector<unsigned> sampled_line_indices;
        for (int i = 0; i<line_sample_num && i< input_camera_line_infinite.size(); i++) {
            int index = rand()%line_num;
            sampled_line_indices.push_back(index);
        }
        std::sort(sampled_line_indices.begin(), sampled_line_indices.end());
        sampled_line_indices.erase(std::unique(sampled_line_indices.begin(), sampled_line_indices.end()), sampled_line_indices.end());
        
        // pick inliers in each hypothese
        for (int i = 0; i<hypotheses.size(); i++) {
            // evaluate the accuracy by checking transformation
            Eigen::Affine3d cur_affine = hypotheses[i].affine_;
            // 2.1 check points
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
            
            // 2.2 check points on line
            if (hypotheses.size() <= joint_opt_num) {
                Eigen::Vector3d translation = cur_affine.translation();
                Eigen::Matrix3d inv_r = cur_affine.linear().inverse();
                
                for (int j = 0; j<sampled_line_indices.size(); j++) {
                    int index = sampled_line_indices[j];  // line index
                    // from world to camera coordiante
                    vector<unsigned int> inlier_index;
                    for (int k = 0; k<input_world_line_pts_group[index].size(); k++) {
                        if (valid_point_on_line[index][k]) {
                            Eigen::Vector3d p = input_world_line_pts_group[index][k];
                            Eigen::Vector3d c_p = inv_r * (p - translation);
                            double dist = input_camera_line_infinite[index].distance(c_p);
                            if (dist < threshold) {
                                inlier_index.push_back(k);
                            }
                        }
                    }
                    if (inlier_index.size() > min_line_support_num) {
                        hypotheses[i].line_inlier_indices_.push_back(index);
                        hypotheses[i].point_on_line_inlier_index_groups_.push_back(inlier_index);
                    }
                    else {
                        hypotheses[i].energy_ += point_on_line_wt * input_world_line_pts_group[index].size();
                    }
                }
            }
            
            
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
            const PoseHypotheseLoss& hypo = hypotheses[i];  // ? can avoid copy
            // number of inliers is larger than minimum configure
            if (hypo.inlier_indices_.size() > 7) {
                vector<Eigen::Vector3d> inlier_camera_pts;
                vector<Eigen::Vector3d> inlier_world_pts;
                vector<Eigen::Matrix3d> inlier_wordl_pts_precision;
                
                for (int j = 0; j<hypo.inlier_indices_.size(); j++) {
                    int index = hypo.inlier_indices_[j];
                    int wld_sub_index = hypo.inlier_candidate_world_pts_indices_[j];
                    inlier_camera_pts.push_back(input_camera_pts[index]);
                    inlier_world_pts.push_back(input_world_pts[index][wld_sub_index]);
                    inlier_wordl_pts_precision.push_back(input_world_pts_precision[index][wld_sub_index]);
                }
                //printf("inlier number %lu\n", inlier_camera_pts.size());
                // only optimize the camera when there is on candidate
                
                
                if (hypotheses.size() >= joint_opt_num) {
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
                    hypotheses[i].line_inlier_indices_.clear();
                    hypotheses[i].point_on_line_inlier_index_groups_.clear();
                }
                else
                {
                    vector<vector< Eigen::Vector3d> >  inlier_world_line_pts_group(hypo.line_inlier_indices_.size());
                    vector<vector< Eigen::Matrix3d> >  inlier_world_line_pts_precision(hypo.line_inlier_indices_.size());
                    vector<std::pair<Eigen::Vector3d, Eigen::Vector3d> > inlier_camera_lines;
               
                    for (int j = 0; j<hypo.line_inlier_indices_.size(); j++) {
                        int line_index = hypo.line_inlier_indices_[j];
                        inlier_camera_lines.push_back(input_camera_lines[line_index]);
                        for (int k = 0; k<hypo.point_on_line_inlier_index_groups_[j].size(); k++) {
                            int point_index = hypo.point_on_line_inlier_index_groups_[j][k];
                            inlier_world_line_pts_group[j].push_back(input_world_line_pts_group[line_index][point_index]);
                            inlier_world_line_pts_precision[j].push_back(input_world_line_pts_precision[line_index][point_index]);
                        }
                    }
                    
                    
                    Eigen::Affine3d refined_pose;
                    
                    bool is_estimated = VnlAlgo::estimateCameraPoseWithUncertainty(inlier_camera_pts,
                                                                                   inlier_world_pts,
                                                                                   inlier_wordl_pts_precision,
                                                                                   
                                                                                   inlier_world_line_pts_group,
                                                                                   inlier_world_line_pts_precision,
                                                                                   inlier_camera_lines,
                                                                                   hypo.affine_,
                                                                                   refined_pose);                    
                   
                    if (is_estimated) {
                        hypotheses[i].affine_ = refined_pose;
                    }
                    hypotheses[i].inlier_indices_.clear();
                    hypotheses[i].inlier_candidate_world_pts_indices_.clear();
                    hypotheses[i].line_inlier_indices_.clear();
                    hypotheses[i].point_on_line_inlier_index_groups_.clear();                    
                }
            }
        }// i
    } // while
    assert(hypotheses.size() == 1);
    
    output_camera_pose = hypotheses.front().affine_;
    
    return true;
}