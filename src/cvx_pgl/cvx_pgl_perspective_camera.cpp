//
//  cvx_pgl_perspective_camera.cpp
//  CalibMeMatching
//
//  Created by jimmy on 2018-02-05.
//  Copyright (c) 2018 Nowhere Planet. All rights reserved.
//

#include "cvx_pgl_perspective_camera.h"
#include <iostream>

// Eigen
#include <Eigen/Geometry>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>


namespace cvx_pgl {
    
    namespace {
    struct PointLineCameraFunctor
    {
        typedef double Scalar;
        
        typedef Eigen::VectorXd InputType;
        typedef Eigen::VectorXd ValueType;
        typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> JacobianType;
        
        enum {
            InputsAtCompileTime = Eigen::Dynamic,
            ValuesAtCompileTime = Eigen::Dynamic
        };
        
        const vector<Vector2d> model_pts_;
        const vector<Vector2d> im_pts_;
        const vector<std::pair<Eigen::Vector2d, Eigen::Vector2d> > model_lines_;
        const vector<Vector2d> im_line_pts_;
        const Eigen::Vector2d pp_;         // principal point
        
        int m_inputs;
        int m_values;
        
        PointLineCameraFunctor(const vector<Vector2d> & model_pts,
                                   const vector<Vector2d> & im_pts,
                                   const vector<std::pair<Eigen::Vector2d, Eigen::Vector2d> > & model_lines,
                                   const vector<Vector2d> & im_line_pts,
                                   const Eigen::Vector2d& pp):
        model_pts_(model_pts),
        im_pts_(im_pts),
        model_lines_(model_lines),
        im_line_pts_(im_line_pts),
        pp_(pp)
        {
            m_inputs = 7;
            m_values = (int)model_pts_.size() * 2 + (int)model_lines_.size();
            assert(m_values >= 7);
        }
        
        
        int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fx) const
        {
            assert(x.size() == 7);
            
            double fl = x[0];
            Eigen::Vector3d rod(x[1], x[2], x[3]);
            Eigen::Vector3d cc(x[4], x[5], x[6]);
            
            calibration_matrix K(fl, pp_);
            cvx_pgl::perspective_camera camera;
            camera.set_calibration(K);
            camera.set_rotation(rod);
            camera.set_camera_center(cc);
            
            // point to point correspondence
            int idx = 0;
            for (int i = 0; i<model_pts_.size(); i++) {
                double x = 0.0;
                double y = 0.0;
                camera.project(model_pts_[i].x(), model_pts_[i].y(), 0.0, x, y);
                fx[idx++] = im_pts_[i].x() - x;
                fx[idx++] = im_pts_[i].y() - y;
            }
            // point to line distance, from image point to projected model line
            for (int i = 0; i<model_lines_.size(); i++) {
                // project model line into image
                Eigen::Vector3d p1(model_lines_[i].first.x(), model_lines_[i].first.y(), 0);
                Eigen::Vector3d p2(model_lines_[i].second.y(), model_lines_[i].second.y(), 0);
                
                double x1 = 0.0, y1 = 0.0, x2 = 0.0, y2 = 0.0;
                camera.project(p1.x(), p1.y(), p1.z(), x1, y1);
                camera.project(p2.x(), p2.y(), p2.z(), x2, y2);
                
                // measure pixel distance in image
                Eigen::Vector2d q1(x1, y1);
                Eigen::Vector2d q2(x2, y2);
                Eigen::ParametrizedLine<double, 2> line = Eigen::ParametrizedLine<double, 2>::Through(q1, q2);
                double dist = line.distance(im_line_pts_[i]);
                fx[idx++] = dist;
            }
            
            return 0;
        }
        
        int inputs() const { return m_inputs; }  // inputs is the dimension of x.
        int values() const { return m_values; }   // "values" is the number of fx and
        
        void getResult(const Eigen::VectorXd &x,
                       cvx_pgl::perspective_camera & camera,
                       double & reproj_error) const
        {
            assert(x.size() == 7);
            double fl = x[0];
            Eigen::Vector3d rod(x[1], x[2], x[3]);
            Eigen::Vector3d cc(x[4], x[5], x[6]);
            
            calibration_matrix K(fl, pp_);
            camera.set_calibration(K);
            camera.set_rotation(rod);
            camera.set_camera_center(cc);
            
            // point to point correspondence
            reproj_error = 0.0;
            for (int i = 0; i<model_pts_.size(); i++) {
                double x = 0.0;
                double y = 0.0;
                camera.project(model_pts_[i].x(), model_pts_[i].y(), 0.0, x, y);
                Eigen::Vector2d q(x, y);
                reproj_error += (q - im_pts_[i]).squaredNorm();
            }
            // point to line distance, from image point to projected model line
            for (int i = 0; i<model_lines_.size(); i++) {
                // project model line into image
                Eigen::Vector3d p1(model_lines_[i].first.x(), model_lines_[i].first.y(), 0);
                Eigen::Vector3d p2(model_lines_[i].second.y(), model_lines_[i].second.y(), 0);
                
                double x1 = 0.0, y1 = 0.0, x2 = 0.0, y2 = 0.0;
                camera.project(p1.x(), p1.y(), p1.z(), x1, y1);
                camera.project(p2.x(), p2.y(), p2.z(), x2, y2);
                
                // measure pixel distance in image
                Eigen::Vector2d q1(x1, y1);
                Eigen::Vector2d q2(x2, y2);
                Eigen::ParametrizedLine<double, 2> line = Eigen::ParametrizedLine<double, 2>::Through(q1, q2);
                double dist2 = line.squaredDistance(im_line_pts_[i]);
                reproj_error += dist2;
            }
            reproj_error = sqrt(reproj_error/(model_pts_.size() + model_lines_.size()));
        }
    };
    }



    double estimateCamera(const vector<Vector2d> & model_pts,
                        const vector<Vector2d> & im_pts,
                        const vector<std::pair<Eigen::Vector2d, Eigen::Vector2d> > & model_lines,
                        const vector<Vector2d> & im_line_pts,
                        const perspective_camera& init_camera,
                        perspective_camera& refined_camera)
    {
        assert(model_pts.size() == im_pts.size());
        assert(model_lines.size() == im_line_pts.size());
        
        int num = (int)model_pts.size() * 2 + (int)model_lines.size();
        if (num < 7) {
            return INT_MAX;
        }
        
        // initial parameter from input
        Eigen::Vector2d pp = init_camera.get_calibration().principal_point();
        double fl = init_camera.get_calibration().focal_length();
        Eigen::Vector3d rod = init_camera.get_rotation().as_rodrigues();
        Eigen::Vector3d cc = init_camera.get_camera_center();
        
        
        PointLineCameraFunctor opt_functor(model_pts, im_pts, model_lines, im_line_pts, pp);
        Eigen::NumericalDiff<PointLineCameraFunctor> numerical_dif_functor(opt_functor);
        Eigen::LevenbergMarquardt<Eigen::NumericalDiff<PointLineCameraFunctor>, double> lm(numerical_dif_functor);
        lm.parameters.ftol = 1e-6;
        lm.parameters.xtol = 1e-6;
        lm.parameters.maxfev = 100;
        
        Eigen::VectorXd x(7);
        x[0] = fl;
        x[1] = rod.x();
        x[2] = rod.y();
        x[3] = rod.z();
        x[4] = cc.x();
        x[5] = cc.y();
        x[6] = cc.z();
        
        {
            double reproj_error;
            opt_functor.getResult(x, refined_camera, reproj_error);
            std::cout<<"initial reprojection error is "<<reproj_error<<std::endl;
        }
        Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(x);
        //std::cout << "status: " << status << std::endl;
        
        double reproj_error;
        opt_functor.getResult(x, refined_camera, reproj_error);       
        
        return reproj_error;
    }
    
    
    bool estimateCameraRANSAC(const vector<std::pair<Eigen::Vector2d, Eigen::Vector2d> > & model_lines,
                              const vector<Vector2d> & im_line_pts,
                              const perspective_camera& init_camera,
                              perspective_camera& refined_camera)
    {
        assert(model_lines.size() == im_line_pts.size());
        const int min_configuation_num = 7;
        int num_iteration = 1024;
        const double dist_threshold = 2.0;
        const double early_stop_ratio = 0.8;
        
        vector<int> index;
        for (int i = 0; i<model_lines.size(); i++) {
            index.push_back(i);
        }
        vector<Vector2d> dummy_model_pts;
        vector<Vector2d> dummy_im_pts;
        int max_inlier_num = 0;
        
        while(num_iteration--) {
            
            // randomly select 7 pairs
            std::random_shuffle(index.begin(), index.end());
            vector<std::pair<Eigen::Vector2d, Eigen::Vector2d> > cur_lines;
            vector<Vector2d> cur_pts;
            cvx_pgl::perspective_camera cur_camera;
            for (int j = 0; j<min_configuation_num; j++) {
                cur_lines.push_back(model_lines[index[j]]);
                cur_pts.push_back(im_line_pts[index[j]]);
            }
            
            // estimate hypothesis
            double reproj_error = cvx_pgl::estimateCamera(dummy_model_pts, dummy_im_pts,
                                                          cur_lines, cur_pts,
                                                          init_camera, cur_camera);
            //printf("reprojection error %f\n", reproj_error);
            
            
            // verify hypothesis
            vector<int> inlier_index;
            for (int j = 0; j<model_lines.size(); j++) {
                Eigen::Vector3d p1(model_lines[j].first.x(), model_lines[j].first.y(), 0);
                Eigen::Vector3d p2(model_lines[j].second.y(), model_lines[j].second.y(), 0);
                double x1 = 0.0, y1 = 0.0, x2 = 0.0, y2 = 0.0;
                cur_camera.project(p1.x(), p1.y(), p1.z(), x1, y1);
                cur_camera.project(p2.x(), p2.y(), p2.z(), x2, y2);
                
                // measure pixel distance in image
                Eigen::Vector2d q1(x1, y1);
                Eigen::Vector2d q2(x2, y2);
                Eigen::ParametrizedLine<double, 2> line = Eigen::ParametrizedLine<double, 2>::Through(q1, q2);
                double dist = line.distance(im_line_pts[j]);
                if (dist < dist_threshold) {
                    inlier_index.push_back(j);
                }
            }
            // record the best one
            if (inlier_index.size() > max_inlier_num && inlier_index.size() > min_configuation_num) {
                max_inlier_num = (int)inlier_index.size();
                
                // re-estimate camera
                vector<std::pair<Eigen::Vector2d, Eigen::Vector2d> > opt_lines;
                vector<Vector2d> opt_pts;
                for (int j = 0; j<inlier_index.size(); j++) {
                    opt_lines.push_back(model_lines[inlier_index[j]]);
                    opt_pts.push_back(im_line_pts[inlier_index[j]]);
                }
                
                cvx_pgl::estimateCamera(dummy_model_pts, dummy_im_pts,
                                        opt_lines, opt_pts,
                                        init_camera, cur_camera);
                refined_camera = cur_camera;
                double inlier_ratio = 1.0*inlier_index.size()/model_lines.size();
                printf("inlier ratio %f\n", inlier_ratio);
                if (inlier_ratio > early_stop_ratio) {
                    break;
                }
            }
            
        }
        
        return max_inlier_num * 2 > model_lines.size(); // at least 50% is inlier
    }
    
}
