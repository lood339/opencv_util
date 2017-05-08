//
//  vpgl_algo.cpp
//  PointLineReloc
//
//  Created by jimmy on 2017-04-11.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "vpgl_algo.h"
#include "vpgl_head_files.h"
#include "vnl_head_files.h"
#include "vgl_head_files.h"
#include <iostream>

using std::cout;
using std::endl;

class optimize_camera_pose_2d_line_point_residual: public vnl_least_squares_function
{
protected:
    const vector<vgl_point_3d<double> > wld_pts_;
    const vector<vgl_point_2d<double> > img_pts_;
    
    const vector<vgl_infinite_line_3d<double> >  wld_lines_;
    const vector<vector< vgl_point_2d<double> > >  img_line_end_pts_;
    const Eigen::Matrix3d camera_matrix_;
    
public:
    optimize_camera_pose_2d_line_point_residual(const vector<vgl_point_3d<double> >& wld_pts,
                                                const vector<vgl_point_2d<double> >& img_pts,
                                                
                                                const vector<vgl_infinite_line_3d<double> >&  wld_lines,
                                                const vector<vector< vgl_point_2d<double> > >&  img_line_end_pts,
                                                const Eigen::Matrix3d& camera_matrix,
                                                const int num_line_pts):
    vnl_least_squares_function(7, (unsigned int)(img_pts.size() * 2 + num_line_pts), no_gradient),
    wld_pts_(wld_pts),
    img_pts_(img_pts),
    wld_lines_(wld_lines),
    img_line_end_pts_(img_line_end_pts),
    camera_matrix_(camera_matrix)
    {
        assert(wld_pts_.size() == img_pts.size());
        assert(wld_lines_.size() == img_line_end_pts_.size());
    }
    
    void f(vnl_vector<double> const &x, vnl_vector<double> &fx)
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
        
        Eigen::Affine3d affine;
        affine.linear() = R;
        affine.translation() = translation;
        
        Eigen::Affine3d camera_p = camera_matrix_ * affine;
        
        vnl_matrix_fixed<double,3,4> camera_3x4;
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c<4; c++) {
                camera_3x4[r][c] = camera_p(r, c);
            }
        }
        vpgl_proj_camera<double> camera(camera_3x4);
        //loop all points
        int idx = 0;
        for (int i = 0; i<wld_pts_.size(); i++) {
            vgl_point_3d<double> p = wld_pts_[i];
            vgl_point_2d<double> proj_p = (vgl_point_2d<double>)camera.project(p);
            
            fx[idx++] = img_pts_[i].x() - proj_p.x();
            fx[idx++] = img_pts_[i].y() - proj_p.y();
        }
       
        // loop all lines
        for (int i = 0; i<wld_lines_.size(); i++) {
            vgl_line_2d<double> img_line = camera.project(wld_lines_[i]);
            for (int j = 0; j<img_line_end_pts_[i].size(); j++) {
                double dist = vgl_distance(img_line, img_line_end_pts_[i][j]);
                fx[idx++] = dist;
            }
            
        }
    }
    
    void getResult(vnl_vector<double> const &x, Eigen::Affine3d & pose)
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
        
        pose.linear() = R;
        pose.translation() = translation;
    }
};



bool VpglAlgo::estimateCameraPose(const vector<Vector2d> & input_img_pts,
                                  const vector<Vector3d> & input_world_pts,
                                  const vector<vector<Vector2d > > & input_image_line_pts,
                                  const vector<Eigen::ParametrizedLine<double, 3> > & input_world_lines,
                                  const Eigen::Matrix3d& camera_matrix,
                                  const Eigen::Affine3d& init_pose,
                                  Eigen::Affine3d& refined_pose)
{
    assert(input_img_pts.size() == input_world_pts.size());
    assert(input_image_line_pts.size() == input_world_lines.size());
    
    // change format of input data
    vector<vgl_point_3d<double> > wld_pts;
    vector<vgl_point_2d<double> > img_pts;
    for (int i = 0; i<input_img_pts.size(); i++) {
        vgl_point_3d<double> p(input_world_pts[i].x(), input_world_pts[i].y(), input_world_pts[i].z());
        vgl_point_2d<double> q(input_img_pts[i].x(), input_img_pts[i].y());
        wld_pts.push_back(p);
        img_pts.push_back(q);
    }
    
    vector<vgl_infinite_line_3d<double> >  wld_lines;
    vector<vector<vgl_point_2d<double> > >  img_line_end_pts(input_image_line_pts.size());
    int num_line_img_pts = 0;
    for (int i = 0; i<input_world_lines.size(); i++) {
        Eigen::Vector3d o = input_world_lines[i].origin();
        Eigen::Vector3d d = input_world_lines[i].direction();
        
        vgl_point_3d<double> p(o.x(), o.y(), o.z());
        vgl_vector_3d<double> direction(d.x(), d.y(), d.z());
        vgl_infinite_line_3d<double> cur_line(p, direction);
        wld_lines.push_back(cur_line);
      
        for (int j = 0; j<input_image_line_pts[i].size(); j++) {
            vgl_point_2d<double> p(input_image_line_pts[i][j].x(), input_image_line_pts[i][j].y());
            img_line_end_pts[i].push_back(p);
            num_line_img_pts++;
        }
    }
    
    // create optimization 
    optimize_camera_pose_2d_line_point_residual residual(wld_pts, img_pts, wld_lines, img_line_end_pts, camera_matrix, num_line_img_pts);
    
    Eigen::Matrix3d R = init_pose.rotation();
    Eigen::Vector3d trs = init_pose.translation(); // translation
    
    Eigen::Quaternion<double> qt(R);
    qt.normalize();
    
    vnl_vector<double> x(7, 0.0);
    x[0] = qt.x();
    x[1] = qt.y();
    x[2] = qt.z();
    x[3] = qt.w();
    x[4] = trs[0];
    x[5] = trs[1];
    x[6] = trs[2];
    
   // cout<<"initial solution "<<x<<endl;
    
    vnl_levenberg_marquardt lmq(residual);
    lmq.set_x_tolerance(0.0001);
    bool is_minimized = lmq.minimize(x);
    
  //  cout<<"refined solution "<<x<<endl;
    if (is_minimized) {
        residual.getResult(x, refined_pose);
     //   lmq.diagnose_outcome();
        //cout<<"initial pose "<<init_pose<<endl;
        //cout<<"refined pose "<<refined_pose<<endl;
        return true;
    }
    lmq.diagnose_outcome();
    return false;
}


Eigen::ParametrizedLine<double, 2> project3DLine(const Eigen::Affine3d & affine,
                                                 const Eigen::ParametrizedLine<double, 3>& input_line)
{
    // create a projection camera
    vnl_matrix_fixed<double,3,4> camera_matrix;
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c<4; c++) {
            camera_matrix[r][c] = affine(r, c);
        }
    }
    vpgl_proj_camera<double> camera(camera_matrix);
    
    // create a 3D line and project to image
    Eigen::Vector3d o = input_line.origin();
    Eigen::Vector3d d = input_line.direction();
    vgl_point_3d<double> p(o.x(), o.y(), o.z());
    vgl_vector_3d<double> direction(d.x(), d.y(), d.z());
    
    vgl_infinite_line_3d<double> world_line(p, direction);
    vgl_line_2d<double> img_line = camera.project(world_line);
   
    vgl_point_2d<double> p1;
    vgl_point_2d<double> p2;
    img_line.get_two_points(p1, p2);
    
    // create a 2D line as in Eigen
    Eigen::Vector2d q1(p1.x(), p1.y());
    Eigen::Vector2d q2(p2.x(), p2.y());
    Eigen::ParametrizedLine<double, 2> line2d = Eigen::ParametrizedLine<double, 2>::Through(q1, q2);
    
    return line2d;
}
