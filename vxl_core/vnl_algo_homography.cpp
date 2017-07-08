//
//  vnl_algo_homography.cpp
//  CalibMeMatching
//
//  Created by jimmy on 2017-05-29.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "vnl_algo_homography.h"


namespace  {
    class vnl_point_line_homo_residual: public vnl_least_squares_function
    {
    protected:
        const vector<Vector2d> src_pts_;
        const vector<Vector2d> dst_pts_;
        const vector<Eigen::ParametrizedLine<double, 2> > src_lines_;
        const vector<Vector2d> dst_line_pts_;
        
        
    public:
        vnl_point_line_homo_residual(const vector<Vector2d> & src_pts,
                                     const vector<Vector2d> & dst_pts,
                                     const vector<Eigen::ParametrizedLine<double, 2> > & src_lines,
                                     const vector<Vector2d> & dst_line_pts,
                                     int constraint_num):
        vnl_least_squares_function(9, constraint_num, no_gradient),
        src_pts_(src_pts),
        dst_pts_(dst_pts),
        src_lines_(src_lines),
        dst_line_pts_(dst_line_pts){
            assert(src_pts_.size() == dst_pts_.size());
            assert(src_lines_.size() == dst_line_pts_.size());
            assert(constraint_num >= 9);
        }
        
        void f(const vnl_vector<double> &x, vnl_vector<double> &fx)
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
        }
        
        void getResult(const vnl_vector<double> &x, Eigen::Matrix3d & result) const
        {
            for (int i = 0; i<9; i++) {
                result(i/3, i%3) = x[i];
            }
            result.normalize();
        }
    };
}



bool VnlAlgoHomography::estimateHomography(const vector<Vector2d> & src_pts,
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
    
    vnl_point_line_homo_residual residual(src_pts, dst_pts, src_lines, dst_line_pts, num);
    vnl_vector<double> x(9, 0.0);
    for (int i = 0; i<x.size(); i++) {
        x[i] = init_homo(i/3, i%3);
    }
    
    vnl_levenberg_marquardt lmq(residual);
    lmq.set_x_tolerance(0.0001);
    
    bool is_minimized = lmq.minimize(x);
    if (is_minimized) {        
        residual.getResult(x, final_homo);
        return true;
    }
    lmq.diagnose_outcome();
    
    return true;
}