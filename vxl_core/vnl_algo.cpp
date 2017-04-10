//
//  vnl_algo.cpp
//  PointLineReloc
//
//  Created by jimmy on 2017-04-08.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "vnl_algo.h"
#include "vgl_head_files.h"
#include <algorithm>


using Eigen::Matrix3d;
using Eigen::Vector3d;


class vnl_point_line_residual: public vnl_least_squares_function
{
protected:
    vector<Vector3d> camera_pts_;
    vector<Vector3d> world_pts_;
    vector<Vector3d> camera_line_start_pts_;
    vector<Vector3d> camera_line_end_pts_;
    vector<Eigen::ParametrizedLine<double, 3> > world_lines_;
    
public:
    vnl_point_line_residual(const vector<Vector3d> & camera_pts,
                            const vector<Vector3d> & world_pts,
                            const vector<Vector3d> & camera_line_start_pts,
                            const vector<Vector3d> & camera_line_end_pts,
                            const vector<Eigen::ParametrizedLine<double, 3> > & world_lines):
    vnl_least_squares_function(7,
                               (unsigned int)(camera_pts.size() + camera_line_start_pts.size() + camera_line_end_pts.size()),
                               no_gradient),
    camera_pts_(camera_pts),
    world_pts_(world_pts),
    camera_line_start_pts_(camera_line_start_pts),
    camera_line_end_pts_(camera_line_end_pts),
    world_lines_(world_lines)
    {
        size_t num = camera_pts.size() + camera_line_start_pts.size() + camera_line_end_pts.size();
        assert(num > 7);
    }
    
    void f(const vnl_vector<double> &x, vnl_vector<double> &fx)
    {
        double qx = x[0];
        double qy = x[1];
        double qz = x[2];
        double qw = x[3];
        
        Eigen::Quaternion<double> qt(qw, qx, qy, qz);
        qt.normalize();
        Eigen::Matrix3d R = qt.toRotationMatrix();
        Eigen::Vector3d translation;   // translation
        translation[0] = x[4];
        translation[1] = x[5];
        translation[2] = x[6];
        
        // point correspondence
        const int N = (int)camera_pts_.size();
        for (int i = 0; i<camera_pts_.size(); i++) {
            Eigen::Vector3d p = camera_pts_[i];
            Eigen::Vector3d q = R * p + translation;
            double dist2 = (q - world_pts_[i]).norm();
            fx[i] = dist2;
        }
        
        // line end point projections
        for (int i =0; i<camera_line_start_pts_.size(); i++) {
            
            {
                Eigen::Vector3d p = camera_line_start_pts_[i];
                Eigen::Vector3d q = R * p + translation;
                double dist2 = world_lines_[i].distance(q);
                fx[N + 2 * i] = dist2;
            }
            
            {
                Eigen::Vector3d p = camera_line_end_pts_[i];
                Eigen::Vector3d q = R * p + translation;
                double dist2 = world_lines_[i].distance(q);
                fx[N + 2 * i + 1] = dist2;
            }
        }
    }
    
    void getResult(const vnl_vector<double> &x, Eigen::Affine3d& final_camera)
    {
        double qx = x[0];
        double qy = x[1];
        double qz = x[2];
        double qw = x[3];
        
        Eigen::Quaternion<double> qt(qw, qx, qy, qz);
        qt.normalize();
        Eigen::Matrix3d R = qt.toRotationMatrix();
        Eigen::Vector3d translation;   // translation
        translation[0] = x[4];
        translation[1] = x[5];
        translation[2] = x[6];
        
        final_camera.linear() = R;
        final_camera.translation() = translation;
    }
   
    
};

bool VnlAlgo::estimateCameraPose(const vector<Vector3d> & camera_pts,
                        const vector<Vector3d> & world_pts,
                        const vector<Vector3d> & camera_line_start_pts,
                        const vector<Vector3d> & camera_line_end_pts,
                        const vector<Eigen::ParametrizedLine<double, 3> > & world_lines,
                        const Eigen::Affine3d& init_camera,
                        Eigen::Affine3d& final_camera)
{
    
    Eigen::Matrix3d R = init_camera.rotation();
    Eigen::Vector3d trs = init_camera.translation(); // translation    
    
    vnl_point_line_residual residual(camera_pts, world_pts, camera_line_start_pts, camera_line_end_pts, world_lines);
    
    Eigen::Quaternion<double> qt(R);
    
    vnl_vector<double> x(7, 0.0);
    x[0] = qt.x();
    x[1] = qt.y();
    x[2] = qt.z();
    x[3] = qt.w();
    x[4] = trs[0];
    x[5] = trs[1];
    x[6] = trs[2];
    
    vnl_levenberg_marquardt lmq(residual);
    lmq.set_x_tolerance(0.0001);
    bool is_minimized = lmq.minimize(x);
    if (is_minimized) {
        residual.getResult(x, final_camera);
        return true;
    }
    lmq.diagnose_outcome();
    return false;
}