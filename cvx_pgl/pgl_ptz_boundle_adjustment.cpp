//
//  pgl_ptz_boundle_adjustment.cpp
//  CalibMeMatching
//
//  Created by jimmy on 2017-08-05.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "pgl_ptz_boundle_adjustment.h"
#include <iostream>

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

using Eigen::VectorXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using std::cout;
using std::endl;

namespace cvx_pgl {
    namespace {
        struct PTZFunctor {
            typedef double Scalar;
            
            typedef Eigen::VectorXd InputType;
            typedef Eigen::VectorXd ValueType;
            typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> JacobianType;
            
            enum {
                InputsAtCompileTime = Eigen::Dynamic,
                ValuesAtCompileTime = Eigen::Dynamic
            };
            
            const vector<vector<Eigen::Vector2d> > image_points_;
            const vector<vector<int> > visibility_;
            const Vector2d pp_;
            const int point_num_;
            
            int m_inputs;
            int m_values;
            
            PTZFunctor(const vector<vector<Eigen::Vector2d> >& image_points,
                       const vector<vector<int> >& visibility,
                       const Vector2d& pp,
                       const int point_num):
            image_points_(image_points),
            visibility_(visibility),
            pp_(pp),
            point_num_(point_num)
            {
                // each camera has three freedom, each point has two freedom
                m_inputs = 3 * (int)image_points_.size() + point_num_ * 2;
                m_values = 0;
                for(int i = 0; i<image_points_.size(); i++) {
                    assert(image_points_[i].size() == visibility_[i].size());
                    m_values += (int)image_points_[i].size() * 2; // each point has two constraint
                }
                assert(m_values > m_inputs);
            }
            
            int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fx) const
            {
                const int ptz_n = (int)image_points_.size();
                
                vector<Vector3d> ptzs; // estiamted pan, tilt and focal length
                for(int i = 0; i<ptz_n; i++) {
                    ptzs.push_back(Vector3d(x[3*i], x[3*i+1], x[3*i+2]));
                }
                
                int start_index = ptz_n * 3;
                int index = 0;
                // loop each image
                for(int i = 0; i<image_points_.size(); i++) {
                    // loop each point
                    for(int j = 0; j<visibility_[i].size(); j++) {
                        int point_index = visibility_[i][j];
                        Vector2d point_pan_tilt(x[start_index + point_index*2 + 0], x[start_index + point_index*2 + 1]); //x to be optimized
                        Vector2d p = image_points_[i][j];  // observation
                        Vector2d q = panTilt2Point(pp_, ptzs[i], point_pan_tilt); // projection
                        
                        fx[index] = q.x() - p.x();
                        index++;
                        fx[index] = q.y() - p.y();
                        index++;
                    }
                }
                assert(index == m_values);
                
                return 0;
            }
            
            void printReprojectionError(const Eigen::VectorXd& x,
                                        Eigen::MatrixXd &mean,
                                        Eigen::MatrixXd &cov,
                                        bool verbose) const
            {
                
                
                const int ptz_n = (int)image_points_.size();
                
                vector<Vector3d> ptzs; // estiamted pan, tilt and focal length
                for(int i = 0; i<ptz_n; i++) {
                    ptzs.push_back(Vector3d(x[3*i], x[3*i+1], x[3*i+2]));
                }
                
                int start_index = ptz_n * 3;
                int index = 0;
                
                Eigen::MatrixXd error(m_values/2, 1);
                // loop each image
                for(int i = 0; i<image_points_.size(); i++) {
                    // loop each point
                    for(int j = 0; j<visibility_[i].size(); j++) {
                        int point_index = visibility_[i][j];
                        Vector2d point_pan_tilt(x[start_index + point_index*2 + 0], x[start_index + point_index*2 + 1]); //x to be optimized
                        Vector2d p = image_points_[i][j];  // observation
                        Vector2d q = panTilt2Point(pp_, ptzs[i], point_pan_tilt); // projection
                        
                        double dx = q.x() - p.x();
                        double dy = q.y() - p.y();
                        
                        error(index, 0) = sqrt(dx*dx + dy*dy);
                        index++;
                    }
                }
                
                mean = error.colwise().mean();
                MatrixXd centered = error.rowwise() - error.colwise().mean();
                cov = (centered.adjoint() * centered) / error.rows();
                
                if(verbose) {
                    cout<<"reprojection error mean: "<<mean<<endl;
                    cout<<"reprojection error covariance: "<<cov<<endl<<endl;
                }
            }

            
            void get_result(const Eigen::VectorXd &x,
                            vector<Eigen::Vector2d>& points,
                            vector<Vector3d> & pan_tilt_fl) const
            {
                assert(2 * points.size() + 3 * pan_tilt_fl.size() == m_inputs);
                
                const int ptz_n = (int)image_points_.size();
                // estiamted pan, tilt and focal length
                for(int i = 0; i<ptz_n; i+=3) {
                    pan_tilt_fl[i] = Vector3d(x[3*i], x[3*i+1], x[3*i+2]);
                }
                int start_index = ptz_n * 3;
                for(int i = 0; i<point_num_; i++) {
                    points[i] = Eigen::Vector2d(x[start_index + 2*i + 0], x[start_index + 2*i + 1]);
                }
            }
            
            int inputs() const { return m_inputs; }  // inputs is the dimension of x.
            int values() const { return m_values; }   // "values" is the number of fx and
                            
        };
        
    }
    
    double pgl_ptz_ba::run(vector<Eigen::Vector2d>& points,
                           const vector<vector<Eigen::Vector2d> >& image_points,
                           const vector<vector<int> >& visibility,
                           const Eigen::Vector2d & pp,
                           vector<Vector3d> & pan_tilt_fl)
    {
        assert(image_points.size() == visibility.size());
        assert(image_points.size() == pan_tilt_fl.size());
        for (int i =0; i<image_points.size(); i++) {
            assert(image_points[i].size() == visibility[i].size());
            assert(visibility[i].size() >= 2);
        }
        
        const int step = 10;
        // set optimizer parameter
        PTZFunctor opt_functor(image_points, visibility, pp, int(points.size()));
        Eigen::NumericalDiff<PTZFunctor> dif_functor(opt_functor);
        Eigen::LevenbergMarquardt<Eigen::NumericalDiff<PTZFunctor>, double> lm(dif_functor);
        lm.parameters.ftol = 1e-6;
        lm.parameters.xtol = 1e-6;
        lm.parameters.maxfev = max_iteration_/step;
        
        // initial values
        Eigen::VectorXd x(3 * image_points.size() + points.size() * 2);
        for (int i = 0; i<pan_tilt_fl.size(); i++) {
            x[3*i + 0] = pan_tilt_fl[i][0];
            x[3*i + 1] = pan_tilt_fl[i][1];
            x[3*i + 2] = pan_tilt_fl[i][2];
        }
        const int start_index = 3 * (int)image_points.size();
        for (int i = 0; i<points.size(); i++) {
            x[start_index + i*2 + 0] = points[i][0];
            x[start_index + i*2 + 1] = points[i][1];
        }
        
        bool is_optimized = false;
        MatrixXd mean, cov;
        for (int i = 0; i<step; i++) {
            Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(x);
            printf("LevenbergMarquardt status %d\n", status);
            
            
            opt_functor.printReprojectionError(x, mean, cov, true);
            
            if (mean(0, 0) < reprojection_error_threshold_) {
                printf("stop at %d iterations.\n", i * max_iteration_/step);
                is_optimized = true;
                opt_functor.get_result(x, points, pan_tilt_fl);
                break;
            }
        }
        
        return mean(0, 0);
    }
    
    bool boundle_adjustment(ptz_camera& camera1, ptz_camera& camera2,
                            MatrixXd& location1,  MatrixXd& location2)
    {
        assert(location1.rows() == location2.rows());
        assert(location1.cols() == 2);
        assert(location2.cols() == 2);
        
        const int N = (int)location1.rows();
        Eigen::Vector2d pp(1280.0/2, 720.0/2);
        
        vector<Eigen::Vector2d> points;
        vector<vector<Eigen::Vector2d> > image_points(2);
        vector<vector<int> > visibility(2);
        vector<Vector3d> pan_tilt_fl;
        
        
        // Step 1: adjust pan and tilt in the spherical space
        
        Vector3d ptz1 = camera1.ptz();
        Vector3d ptz2 = camera2.ptz();
        for (int i =0 ; i<N; i++) {
            Eigen::Vector2d point1 = location1.row(i);
            Eigen::Vector2d point2 = location2.row(i);
            Eigen::Vector2d pt1 = point2PanTilt(pp, ptz1, point1);
            Eigen::Vector2d pt2 = point2PanTilt(pp, ptz2, point2);
            Eigen::Vector2d avg_pt = ( pt1 + pt2)/2.0;
            
            points.push_back(avg_pt);
            
            image_points[0].push_back(point1);
            image_points[1].push_back(point2);
            
            visibility[0].push_back(i);
            visibility[1].push_back(i);
        }
        pan_tilt_fl.push_back(ptz1);
        pan_tilt_fl.push_back(ptz2);
        
        
        pgl_ptz_ba ba;
        ba.set_reprojection_error_criteria(2.8);
        ba.run(points, image_points, visibility, pp, pan_tilt_fl);
        
        return true;
    }
}