//
//  cvx_gl_homography.cpp
//  CalibMeMatching
//
//  Created by jimmy on 2017-07-30.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "cvx_gl_homography.h"
#include <iostream>

// Eigen
#include <Eigen/Geometry>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

using std::cout;
using std::endl;

namespace cvx_gl {
    
    namespace  {
        // @brief estimate homography from single-image correspondences
        struct PointLineHomographyFunctor
        {
            typedef double Scalar;
            
            typedef Eigen::VectorXd InputType;
            typedef Eigen::VectorXd ValueType;
            typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> JacobianType;
            
            enum {
                InputsAtCompileTime = Eigen::Dynamic,
                ValuesAtCompileTime = Eigen::Dynamic
            };
            
            const vector<Vector2d> src_pts_;
            const vector<Vector2d> dst_pts_;
            const vector<Eigen::ParametrizedLine<double, 2> > src_lines_;
            const vector<Vector2d> dst_line_pts_;
            
            int m_inputs;
            int m_values;
            
            PointLineHomographyFunctor(const vector<Vector2d> & src_pts,
                                       const vector<Vector2d> & dst_pts,
                                       const vector<Eigen::ParametrizedLine<double, 2> > & src_lines,
                                       const vector<Vector2d> & dst_line_pts):
            src_pts_(src_pts),
            dst_pts_(dst_pts),
            src_lines_(src_lines),
            dst_line_pts_(dst_line_pts)
            
            {
                m_inputs = 9;
                m_values = (int)src_pts.size() * 2 + (int)src_lines.size();
                assert(m_values >= 9);
            }
            
            
            int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fx) const
            {
                Eigen::Matrix3d h;
                for (int i = 0; i<9; i++) {
                    h(i/3, i%3) = x[i];
                }
                h.normalize();
                Eigen::Matrix3d inv_h = h.inverse();
                
                // point to point correspondence
                int idx = 0;
                for (int i = 0; i<src_pts_.size(); i++) {
                    Eigen::Vector3d p(src_pts_[i].x(), src_pts_[i].y(), 1.0);
                    Eigen::Vector3d q = h * p;
                    assert(q.z() != 0);
                    q /= q.z();
                    
                    fx[idx++] = dst_pts_[i].x() - q.x();
                    fx[idx++] = dst_pts_[i].y() - q.y();
                }
                
                // point to line distance, from destination to source
                for (int i = 0; i<src_lines_.size(); i++) {
                    Eigen::Vector3d p(dst_line_pts_[i].x(), dst_line_pts_[i].y(), 1.0);
                    Eigen::Vector3d q = inv_h * p;
                    assert(q.z() != 0);
                    q /= q.z();
                    
                    double dist = src_lines_[i].distance(Eigen::Vector2d(q.x(), q.y()));
                    fx[idx++] = dist;
                }
                return 0;
            }
            
            int inputs() const { return m_inputs; }  // inputs is the dimension of x.
            int values() const { return m_values; }   // "values" is the number of fx and
            
            void getResult(const Eigen::VectorXd &x, Eigen::Matrix3d & result) const
            {
                for (int i = 0; i<9; i++) {
                    result(i/3, i%3) = x[i];
                }
                result.normalize();
            }
        };
    }
    
    
    bool estimateHomography(const vector<Vector2d> & src_pts,
                            const vector<Vector2d> & dst_pts,
                            const vector<Eigen::ParametrizedLine<double, 2> > & src_lines,
                            const vector<Vector2d> & dst_line_pts,
                            const Eigen::Matrix3d& init_homo,
                            Eigen::Matrix3d& final_homo)
    {
        assert(src_pts.size() == dst_pts.size());
        assert(src_lines.size() == dst_line_pts.size());
        
        int num = (int)src_pts.size() * 2 + (int)src_lines.size();
        if (num <= 9) {
            return false;
        }
        
        PointLineHomographyFunctor opt_functor(src_pts, dst_pts, src_lines, dst_line_pts);
        Eigen::NumericalDiff<PointLineHomographyFunctor> numerical_dif_functor(opt_functor);
        Eigen::LevenbergMarquardt<Eigen::NumericalDiff<PointLineHomographyFunctor>, double> lm(numerical_dif_functor);
        lm.parameters.ftol = 1e-6;
        lm.parameters.xtol = 1e-6;
        lm.parameters.maxfev = 100;
        
        Eigen::VectorXd x(9);
        for (int i = 0; i<9; i++) {
            x[i] = init_homo(i/3, i%3);
        }
        
        Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(x);
        std::cout << "status: " << status << std::endl;
        
        opt_functor.getResult(x, final_homo);
        
        return true;
    }
    
    namespace {
        
        struct MultiFramePointLineHomographyFunctor {
            typedef double Scalar;
            typedef Eigen::VectorXd InputType;
            typedef Eigen::VectorXd ValueType;
            typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> JacobianType;
            
            enum {
                InputsAtCompileTime = Eigen::Dynamic,
                ValuesAtCompileTime = Eigen::Dynamic
            };
            
            const vector<vector<Vector2d> > src_pts_;
            const vector<vector<Vector2d> > dst_pts_;
            const vector<vector<Eigen::ParametrizedLine<double, 2> > > src_lines_;
            const vector<vector<Vector2d> > dst_line_pts_;
            const vector<Eigen::Matrix3d> model_to_image_homos_;
            
            int m_inputs;
            int m_values;
            
            MultiFramePointLineHomographyFunctor(const vector<vector<Vector2d> > & src_pts,
                                                 const vector<vector<Vector2d> > & dst_pts,
                                                 const vector<vector<Eigen::ParametrizedLine<double, 2> > > & src_lines,
                                                 const vector<vector<Vector2d> > & dst_line_pts,
                                                 const vector<Eigen::Matrix3d>& model_to_image_homos):
            src_pts_(src_pts),
            dst_pts_(dst_pts),
            src_lines_(src_lines),
            dst_line_pts_(dst_line_pts),
            model_to_image_homos_(model_to_image_homos)
            {
                m_inputs = 9;
                m_values = 0;//(int)src_pts.size() * 2 + (int)src_lines.size();
                assert(src_pts_.size() == dst_pts_.size());
                assert(src_pts_.size() == src_lines_.size());
                assert(src_lines_.size() == dst_line_pts_.size());
                assert(src_pts_.size() == model_to_image_homos_.size());
                
                // point-to-point provides 2 constraint
                // point-on-line provides 1 constraint
                for (int i = 0; i<src_pts_.size(); i++) {
                    m_values += (int)src_pts_[i].size() * 2 + (int)src_lines_[i].size();
                }
                assert(m_values >= 9);
            }
            
            int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fx) const
            {
                Eigen::Matrix3d h;
                for (int i = 0; i<9; i++) {
                    h(i/3, i%3) = x[i];
                }
                h.normalize();
                
                // point to point correspondence
                int idx = 0;
                for (int i = 0; i<src_pts_.size(); i++) {
                    // source image --> model --> destination image
                    Eigen::Matrix3d cur_h = h * model_to_image_homos_[i].inverse();
                    for (int j = 0; j<src_pts_[i].size(); j++) {
                        Eigen::Vector3d p(src_pts_[i][j].x(), src_pts_[i][j].y(), 1.0);
                        Eigen::Vector3d q = cur_h * p;
                        assert(q.z() != 0);
                        q /= q.z();
                        
                        fx[idx++] = dst_pts_[i][j].x() - q.x();
                        fx[idx++] = dst_pts_[i][j].y() - q.y();
                    }
                }
                
                // point to line distance, from destination to source
                for (int i = 0; i<src_lines_.size(); i++) {
                    // source image --> model --> destination image
                    // note from destination to source
                    Eigen::Matrix3d cur_h_inv = (h * model_to_image_homos_[i].inverse()).inverse();
                    for (int j = 0; j<src_lines_[i].size(); j++) {
                        Eigen::Vector3d p(dst_line_pts_[i][j].x(), dst_line_pts_[i][j].y(), 1.0);
                        Eigen::Vector3d q = cur_h_inv * p;
                        assert(q.z() != 0);
                        q /= q.z();
                        
                        double dist = src_lines_[i][j].distance(Eigen::Vector2d(q.x(), q.y()));
                        fx[idx++] = dist;
                    }
                }
                return 0;
            }
            
            int inputs() const { return m_inputs; }  // inputs is the dimension of x.
            int values() const { return m_values; }   // "values" is the number of fx and
            
            void getResult(const Eigen::VectorXd &x, Eigen::Matrix3d & result) const
            {
                for (int i = 0; i<9; i++) {
                    result(i/3, i%3) = x[i];
                }
                result.normalize();
            }
        };
    }
    
    bool estimateHomography(const vector<vector<Vector2d> > & src_pts,
                            const vector<vector<Vector2d> > & dst_pts,
                            const vector<vector<Eigen::ParametrizedLine<double, 2> > > & src_lines,
                            const vector<vector<Vector2d> > & dst_line_pts,
                            const vector<Eigen::Matrix3d>& model_to_image_homos,
                            const Eigen::Matrix3d& init_homo,
                            Eigen::Matrix3d& refined_homo)
    {
        assert(src_pts.size() == dst_pts.size());
        assert(src_lines.size() == dst_line_pts.size());
        
        int num = 0;
        for (int i = 0; i<src_pts.size(); i++) {
            num += (int)src_pts[i].size() * 2 + (int)src_lines[i].size();
        }
        if (num <= 9) {
            return false;
        }
        
        MultiFramePointLineHomographyFunctor opt_functor(src_pts, dst_pts, src_lines, dst_line_pts, model_to_image_homos);
        Eigen::NumericalDiff<MultiFramePointLineHomographyFunctor> numerical_dif_functor(opt_functor);
        Eigen::LevenbergMarquardt<Eigen::NumericalDiff<MultiFramePointLineHomographyFunctor>, double> lm(numerical_dif_functor);
        lm.parameters.ftol = 1e-6;
        lm.parameters.xtol = 1e-6;
        lm.parameters.maxfev = 100;
        
        
        Eigen::VectorXd x(9);
        for (int i = 0; i<9; i++) {
            x[i] = init_homo(i/3, i%3);
        }
        
        Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(x);
        std::cout << "status: " << status << std::endl;
        
        opt_functor.getResult(x, refined_homo);
        return true;
    }
    
} // namespace
