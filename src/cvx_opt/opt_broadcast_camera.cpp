//
//  opt_broadcast_camera.cpp
//  EstimatePTZTripod
//
//  Created by jimmy on 11/15/18.
//  Copyright (c) 2018 Nowhere Planet. All rights reserved.
//

#include "opt_broadcast_camera.h"

// Eigen
#include <Eigen/Geometry>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

#include "pgl_broadcast_camera.h"

namespace cvx {
    namespace  {
        // @brief
        struct CommomCameraCenterDisplacementAndRotationFunctor
        {
            typedef double Scalar;
            
            typedef Eigen::VectorXd InputType;
            typedef Eigen::VectorXd ValueType;
            typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> JacobianType;
            
            enum {
                InputsAtCompileTime = Eigen::Dynamic,
                ValuesAtCompileTime = Eigen::Dynamic
            };
            
            const vector<vector<Vector3d>> wld_pts_; // world coordinate
            const vector<vector<Vector2d>> img_pts_; // image coordinate
            const Vector2d pp_; // principal point
            
            int param_num_;  // number of unknow parameters
            int constraint_num_;  // number of constraint
            
            CommomCameraCenterDisplacementAndRotationFunctor(const vector<vector<Vector3d>> & wld_pts,
                                                             const vector<vector<Vector2d>> & img_pts,
                                                             const Vector2d & pp):
            wld_pts_(wld_pts),
            img_pts_(img_pts),
            pp_(pp)
            {
                assert(wld_pts.size() == img_pts.size());
                param_num_ = 6 + 6 + 3 * (int)wld_pts.size();
                constraint_num_ = 0;
                for(const auto& item: wld_pts) {
                    constraint_num_ += (int)item.size() * 2;
                }
                assert(constraint_num_ >= param_num_);
            }
            
            
            int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fx) const
            {
                Eigen::Vector3d cc(x[0], x[1], x[2]);
                Eigen::Vector3d rod(x[3], x[4], x[5]);
                Eigen::VectorXd lambda = x.segment(6, 6);
                
                cvx::broadcast_camera camera(pp_, cc, rod, lambda);
                int idx = 0;
                for(int i = 0; i<wld_pts_.size(); i++) {
                    double pan  = x[12 + 3 * i + 0];
                    double tilt = x[12 + 3 * i + 1];
                    double fl   = x[12 + 3 * i + 2];
                    
                    Vector3d ptz(pan, tilt, fl);
                    camera.set_ptz(ptz);
                    
                    // point to point correspondence
                    for(int j = 0; j<wld_pts_[i].size(); j++) {
                        Vector2d p = img_pts_[i][j];
                        Vector2d q = camera.project3d(wld_pts_[i][j]);
                        fx[idx++] = p.x() - q.x();
                        fx[idx++] = p.y() - q.y();
                    }
                }
                return 0;
            }
            
            int inputs() const { return param_num_; }  // inputs is the dimension of x.
            int values() const { return constraint_num_; }   // "values" is the number of fx and
            
            void getResult(const Eigen::VectorXd &x,
                           vector<broadcast_camera>& cameras,
                           Eigen::VectorXd& reprojection_errors) const
            {
                Eigen::Vector3d cc(x[0], x[1], x[2]);
                Eigen::Vector3d rod(x[3], x[4], x[5]);
                Eigen::VectorXd lambda = x.segment(6, 6);
                
                cameras.clear();
                cvx::broadcast_camera camera(pp_, cc, rod, lambda);
                for(int i = 0; i<wld_pts_.size(); i++) {
                    double pan  = x[12 + 3 * i + 0];
                    double tilt = x[12 + 3 * i + 1];
                    double fl   = x[12 + 3 * i + 2];
                    
                    
                    Vector3d ptz(pan, tilt, fl);
                    camera.set_ptz(ptz);
                    
                    // point to point correspondence
                    double error = 0.0;
                    for(int j = 0; j<wld_pts_[i].size(); j++) {
                        Vector2d p = img_pts_[i][j];
                        Vector2d q = camera.project3d(wld_pts_[i][j]);
                        double dx = p.x() - q.x();
                        double dy = p.y() - q.y();
                        error += sqrt(dx * dx + dy * dy);
                        //printf("dx dy: %f %f\n", dx, dy);
                    }
                    //printf("\n");
                    error /= wld_pts_[i].size();
                    reprojection_errors[i] = error;
                    cameras.push_back(camera);
                }
                assert(cameras.size() == reprojection_errors.size());
            }
        };
    }

    
    bool estimateBroadcastCameras(const vector<vector<Vector3d>> & wld_pts,
                                  const vector<vector<Vector2d>> & img_pts,
                                  const vector<perspective_camera> & init_cameras,
                                  const Vector3d & init_common_rotation,
                                  vector<broadcast_camera> & estimated_cameras)
    {
        // step 1: check input
        assert(wld_pts.size() == img_pts.size());
        assert(wld_pts.size() == init_cameras.size());
        const int N = (int)init_cameras.size();
        for (int i = 0; i<N; i++) {
            if(wld_pts[i].size() < 3) {
                printf("Error: at least 3 correspondences for a camera to robust estimate PTZ.\n");
                return false;
            }
            assert(wld_pts[i].size() == img_pts[i].size());
        }
        
        const double error_threshold = 3.0; // pixel
        using ResidualFunctor = CommomCameraCenterDisplacementAndRotationFunctor;
        Vector2d pp = init_cameras[0].get_calibration().principal_point();
        
        // step 2: prepare data
        Eigen::VectorXd x(6 + 6 + 3 * (int)init_cameras.size()); // 6 + 6 + 3 * N, pan, tilt, focal length
        x.setZero();
        
        Matrix3d Rs_inv = cvx::rotation_3d(init_common_rotation).as_matrix().inverse();
        
        for (int i = 0; i<init_cameras.size(); i++) {
            perspective_camera cur_camera = init_cameras[i];
            double fl = cur_camera.get_calibration().focal_length();
            
            // R_pan_tilt * Rs = R--> R_pt = R * inv(Rs)
            Eigen::Matrix3d R_pan_tilt = cur_camera.get_rotation().as_matrix() * Rs_inv;
            double cos_pan = R_pan_tilt(0, 0);
            double sin_pan = -R_pan_tilt(0, 2);
            double cos_tilt = R_pan_tilt(1, 1);
            double sin_tilt = -R_pan_tilt(2, 1);
            double pan  = atan2(sin_pan, cos_pan) * 180.0 /M_PI;
            double tilt = atan2(sin_tilt, cos_tilt) * 180.0 /M_PI;
            x[12 + 3 * i + 0] = pan;
            x[12 + 3 * i + 1] = tilt;
            x[12 + 3 * i + 2] = fl;
            //printf("pan, tilt, focal length: %f %f %f\n", pan, tilt, fl);
            
            x[0] += cur_camera.get_camera_center().x();
            x[1] += cur_camera.get_camera_center().y();
            x[2] += cur_camera.get_camera_center().z();
        }
        x[0] /= N;
        x[1] /= N;
        x[2] /= N;
        x[3] = init_common_rotation.x();
        x[4] = init_common_rotation.y();
        x[5] = init_common_rotation.z();
        
        
        ResidualFunctor opt_functor(wld_pts,
                                    img_pts,
                                    pp);
        Eigen::NumericalDiff<ResidualFunctor> numerical_dif_functor(opt_functor);
        Eigen::LevenbergMarquardt<Eigen::NumericalDiff<ResidualFunctor>, double> lm(numerical_dif_functor);
        lm.parameters.ftol = 1e-6;
        lm.parameters.xtol = 1e-6;
        lm.parameters.maxfev = 100;
        
        if (0)
        {
            // debug
            Eigen::VectorXd errors(N);
            opt_functor.getResult(x,
                                  estimated_cameras,
                                  errors);
            std::cout<<"Debug: initial reprojection error: "<<errors.transpose()<<std::endl;            
        }
        
        Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(x);
        //std::cout<<status<<std::endl;
        
        Eigen::VectorXd errors(N);
        opt_functor.getResult(x,
                              estimated_cameras,
                              errors);
        // check reprojection error
        double max_reprojection_error = errors.maxCoeff();
        if (max_reprojection_error > error_threshold) {
            std::cout<<"Warning, large reprojection error: "<<errors.transpose()<<std::endl;
        }
        else {
            std::cout<<"Small reprojection error: "<<errors.transpose()<<std::endl;
        }
        assert(estimated_cameras.size() == N);
      
        return max_reprojection_error < error_threshold;
    }
}