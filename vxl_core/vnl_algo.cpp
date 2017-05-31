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
#include "cvx_line.h"


using Eigen::Matrix3d;
using Eigen::Vector3d;
using std::cout;
using std::endl;


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
        assert(num >= 7);
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
    
    int num = (int)camera_pts.size() + (int)camera_line_start_pts.size() + (int)camera_line_end_pts.size();
    if (num < 7) {
        return false;
    }
    
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


class vnl_line3d_residual: public vnl_least_squares_function
{
protected:
    vector< vector<Vector3d> > camera_pts_;
    vector<Eigen::ParametrizedLine<double, 3> > world_lines_;
    
public:
    vnl_line3d_residual(const vector< vector<Vector3d> > & camera_pts,
                        const vector<Eigen::ParametrizedLine<double, 3> > & world_lines,
                        const int pts_num):
    vnl_least_squares_function(7, pts_num, no_gradient)
    {
        assert(camera_pts.size() == world_lines.size());        
        for (int i = 0; i<camera_pts.size(); i++) {
            camera_pts_.push_back(camera_pts[i]);
            world_lines_.push_back(world_lines[i]);
        }
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
        
        // project from camera coordinate to world coordinate
        int index = 0;
        for (int i = 0; i<world_lines_.size(); i++) {
            for (int j = 0; j<camera_pts_[i].size(); j++) {
                Eigen::Vector3d p = R * camera_pts_[i][j] + translation;
                
                double dist = world_lines_[i].distance(p);  // distance from a point to a 3D line                
                fx[index++] = dist;
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

bool VnlAlgo::estimateCameraPoseFromLine(const vector< vector<Vector3d> > & camera_pts,
                                         const vector<Eigen::ParametrizedLine<double, 3> > & world_lines,
                                         const Eigen::Affine3d& init_pose,
                                         Eigen::Affine3d& final_pose)
{
    assert(camera_pts.size() == world_lines.size());
    
    int num_pts = 0;
    for (int i = 0; i<camera_pts.size(); i++) {
        num_pts += camera_pts[i].size();
    }
    if (num_pts < 7) {
        return false;
    }
    
    Eigen::Matrix3d R = init_pose.rotation();
    Eigen::Vector3d trs = final_pose.translation(); // translation
    vnl_line3d_residual residual(camera_pts, world_lines, num_pts);
    
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
        residual.getResult(x, final_pose);
        //lmq.diagnose_outcome();
        return true;
    }
    //lmq.diagnose_outcome();
    return false;
}

class vnl_point_line3d_residual: public vnl_least_squares_function
{
protected:
    vector<Vector3d> source_pts_;
    vector<Vector3d> target_pts_;
    vector< vector<Vector3d> > camera_line_pts_;
    vector<Eigen::ParametrizedLine<double, 3> > world_lines_;
    
public:
    vnl_point_line3d_residual(const vector<Vector3d>& source_pts,
                              const vector<Vector3d>& target_pts,
                              const vector< vector<Vector3d> > & camera_line_pts,
                              const vector<Eigen::ParametrizedLine<double, 3> > & world_lines,
                              const int pts_num):
    vnl_least_squares_function(7, pts_num + (int)source_pts.size(), no_gradient),
    source_pts_(source_pts),
    target_pts_(target_pts)
    {
        assert(source_pts.size() == target_pts.size());
        assert(camera_line_pts.size() == world_lines.size());
        
        for (int i = 0; i<camera_line_pts.size(); i++) {
            camera_line_pts_.push_back(camera_line_pts[i]);
            world_lines_.push_back(world_lines[i]);
        }
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
        
        // project from camera coordinate to world coordinate
        int index = 0;
        for (int i = 0; i<source_pts_.size(); i++) {
            Eigen::Vector3d p = R * source_pts_[i] + translation;  // rotate and translate a source point
            double dist = (p - target_pts_[i]).norm();   // distance from a point to a target point
            fx[index++] = dist;
        }
        
        for (int i = 0; i<world_lines_.size(); i++) {
            for (int j = 0; j<camera_line_pts_[i].size(); j++) {
                Eigen::Vector3d p = R * camera_line_pts_[i][j] + translation;
                
                double dist = world_lines_[i].distance(p);  // distance from a point to a 3D line
                fx[index++] = dist;
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

bool VnlAlgo::estimateCameraPoseUsingPointLine(const vector<Vector3d>& source_pts,
                                               const vector<Vector3d>& target_pts,
                                               const vector< vector<Vector3d> > & source_line_pts,
                                               const vector<Eigen::ParametrizedLine<double, 3> > & target_lines,
                                               const Eigen::Affine3d& init_pose,
                                               Eigen::Affine3d& final_pose)
{
    assert(source_pts.size() == target_pts.size());
    assert(source_line_pts.size() == target_lines.size());
    
    int num_pts = 0;
    for (int i = 0; i<source_line_pts.size(); i++) {
        num_pts += source_line_pts[i].size();
    }
    if (num_pts + source_pts.size() < 7) {
        return false;
    }
    
    Eigen::Matrix3d R = init_pose.rotation();
    Eigen::Vector3d trs = init_pose.translation(); // translation
    vnl_point_line3d_residual residual(source_pts, target_pts, source_line_pts, target_lines, num_pts);
    
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
        residual.getResult(x, final_pose);
        //lmq.diagnose_outcome();
        return true;
    }
    //lmq.diagnose_outcome();
    return false;
}


class vnl_point_mahalanobis_residual: public vnl_least_squares_function
{
protected:
    vector<Vector3d> source_pts_;
    vector<Vector3d> target_pts_;
    vector<Matrix3d> target_covariance_inv_;
    
public:
    vnl_point_mahalanobis_residual(const vector<Vector3d>& source_pts,
                                   const vector<Vector3d>& target_pts,
                                   const vector<Matrix3d> & target_covariance_inv):
    vnl_least_squares_function(7, (int)source_pts.size(), no_gradient),
    source_pts_(source_pts),
    target_pts_(target_pts),
    target_covariance_inv_(target_covariance_inv)
    {
        assert(source_pts.size() == target_pts.size());
        assert(target_covariance_inv.size() == target_pts.size());
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
        
        // project from camera coordinate to world coordinate
        for (int i = 0; i<source_pts_.size(); i++) {
            Eigen::Vector3d p = R * source_pts_[i] + translation;  // rotate and translate a source point
            Eigen::Vector3d dif = p - target_pts_[i];
            
            // mahalanobis distance
            double dist = dif.transpose() * target_covariance_inv_[i] * dif;
            if (std::isnan(dist) || dist < 0.0) {
                /*
                cout<<"target precision matrix: \n"<<target_covariance_inv_[i]<<endl;
                cout<<"mahalanobis distance is "<<dist<<endl;                
                cout<<"target covariance matrix: \n"<<cov<<endl;
                 */
                Eigen::Matrix3d cov = target_covariance_inv_[i].inverse();
                double det = cov.determinant();
                cout<<"negative mahalanobis distance, determinant is "<<det<<endl;
            }
            else {
                //Eigen::Matrix3d cov = target_covariance_inv_[i].inverse();
                //double det = cov.determinant();
                //cout<<"positive mahalanobis distance, determinant is "<<det<<endl;
            }
            dist = std::max(0.0, dist); // approximate the distance if mahalanobis distance is negative
            dist = sqrt(dist + 0.0000001);
            assert(dist >= 0.0);
            //printf("mahalanobis distance is %lf \n", dist);
            fx[i] = dist;
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


bool VnlAlgo::estimateCameraPoseWithUncertainty(const vector<Eigen::Vector3d>& source_pts, // camera points
                                                const vector<Eigen::Vector3d>& target_pts, // world coordiante points (predicted value)
                                                const vector<Eigen::Matrix3d>& target_covariance_inv, // world coordinate points covariance
                                                const Eigen::Affine3d& init_pose,
                                                Eigen::Affine3d& refined_pose)
{
    assert(source_pts.size() == target_pts.size());
    assert(target_covariance_inv.size() == target_pts.size());
    
    if (source_pts.size() <= 7) {
        return false;
    }
    
    // check invert matrix
    for (int i = 0; i<target_covariance_inv.size(); i++) {
        if (target_covariance_inv[i].hasNaN()) {
            return false;
        }
    }
    
    Eigen::Matrix3d R = init_pose.rotation();
    Eigen::Vector3d trs = init_pose.translation(); // translation
    vnl_point_mahalanobis_residual residual(source_pts, target_pts, target_covariance_inv);
    
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
        //lmq.diagnose_outcome();
        residual.getResult(x, refined_pose);
        return true;
    }
    //lmq.diagnose_outcome();
    return false;
}


class vnl_point_line_mahalanobis_residual: public vnl_least_squares_function
{
protected:
    vector<Vector3d> source_pts_;
    vector<Vector3d> target_pts_;
    vector<Matrix3d> target_covariance_inv_;
    
    vector<vector< Eigen::Vector3d> >  world_line_pts_group_;
    vector<vector< Eigen::Matrix3d> > world_line_pts_precision_;
    vector<std::pair<Eigen::Vector3d, Eigen::Vector3d> > camera_lines_;
    vector<Eigen::ParametrizedLine<double, 3> > camera_lines_infinite_;
    
public:
    vnl_point_line_mahalanobis_residual(const vector<Vector3d>& source_pts,
                                        const vector<Vector3d>& target_pts,
                                        const vector<Matrix3d> & target_covariance_inv,
                                        
                                        const vector<vector< Eigen::Vector3d> > & world_line_pts_group,
                                        const vector<vector< Eigen::Matrix3d> >& world_line_pts_precision,
                                        const vector<std::pair<Eigen::Vector3d, Eigen::Vector3d> >& camera_lines,
                                        const int num_constraint):
    vnl_least_squares_function(7, num_constraint, no_gradient),
    source_pts_(source_pts),
    target_pts_(target_pts),
    target_covariance_inv_(target_covariance_inv),
    world_line_pts_group_(world_line_pts_group),
    world_line_pts_precision_(world_line_pts_precision),
    camera_lines_(camera_lines)
    {
        assert(num_constraint > 7);
        assert(source_pts.size() == target_pts.size());
        assert(target_covariance_inv.size() == target_pts.size());
        
        assert(world_line_pts_group.size() == world_line_pts_precision.size());
        assert(world_line_pts_group.size() == camera_lines.size());
        for (unsigned i = 0; i<camera_lines_.size(); i++) {
            Eigen::ParametrizedLine<double, 3> line = Eigen::ParametrizedLine<double, 3>::Through(camera_lines_[i].first, camera_lines_[i].second);
            camera_lines_infinite_.push_back(line);
        }
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
        
        // project from camera coordinate to world coordinate
        int index = 0;
        // 1. point constraint
        for (unsigned i = 0; i<source_pts_.size(); i++) {
            Eigen::Vector3d p = R * source_pts_[i] + translation;  // rotate and translate a source point, from camera to world
            Eigen::Vector3d dif = p - target_pts_[i];
            // mahalanobis distance
            double dist = dif.transpose() * target_covariance_inv_[i] * dif;
            if (std::isnan(dist) || dist < 0.0) {
                Eigen::Matrix3d cov = target_covariance_inv_[i].inverse();
                double det = cov.determinant();
                cout<<"point negative mahalanobis distance, determinant is "<<det<<endl;
            }
            dist = std::max(0.0, dist); // approximate the distance if mahalanobis distance is negative
            dist = sqrt(dist + 0.0000001);
            assert(dist >= 0.0);
         //   printf("point to point mahalanobis distance is %lf \n", dist);
            fx[index] = dist;
            index++;
        }
        
        Eigen::Matrix3d inv_r = R.inverse();
        
        // 2. line constraint
        for (unsigned i = 0; i<camera_lines_infinite_.size(); i++)  {
            for (unsigned j = 0; j<world_line_pts_group_[i].size(); j++) {
                Eigen::Vector3d p = world_line_pts_group_[i][j];  // world coordinate
                // change from world to camera coordiante
                Eigen::Vector3d c_p = inv_r * (p - translation);  // camera coordinate
                Eigen::Vector3d q = camera_lines_infinite_[i].projection(c_p); //@ omit lambda when calcuate q
                Eigen::Vector3d delta = c_p - q;
                // C' = RCR^T
                Eigen::Matrix3d lambda = inv_r * world_line_pts_precision_[i][j] * R;
                //Eigen::Matrix3d lambda = world_line_pts_precision_[i][j];
                
                double dist = delta.transpose() * lambda * delta;
                if (std::isnan(dist) || dist < 0.0) {
                    Eigen::Matrix3d cov = world_line_pts_precision_[i][j].inverse();
                    double det = cov.determinant();
                    cout<<"line negative mahalanobis distance, determinant is "<<det<<endl;
                    cout<<"covariance matrix \n"<<cov<<endl;
                }                 
                //double dist = delta.norm();
                //printf("point to line mahalanobis distance is %lf %lf\n", dist, delta.norm());
                dist = std::max(0.0, dist); // approximate the distance if mahalanobis distance is negative
                dist = sqrt(dist + 0.0000001);
                assert(dist >= 0.0);
                fx[index] = dist;
                index++;
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


bool VnlAlgo::estimateCameraPoseWithUncertainty(const vector<Eigen::Vector3d>& source_pts, // camera points
                                                const vector<Eigen::Vector3d>& target_pts, // world coordiante points (predicted value)
                                                const vector<Eigen::Matrix3d>& target_covariance_inv, // world coordinate points covariance
                                                // line
                                                const vector<vector< Eigen::Vector3d> > & world_line_pts_group,
                                                const vector<vector< Eigen::Matrix3d> > & world_line_pts_precision,
                                                const vector<std::pair<Eigen::Vector3d, Eigen::Vector3d> >& camera_lines,
                                                
                                                const Eigen::Affine3d& init_pose,
                                                Eigen::Affine3d& refined_pose)
{
    
    assert(source_pts.size() == target_pts.size());
    assert(target_covariance_inv.size() == target_pts.size());
    assert(world_line_pts_group.size() == world_line_pts_group.size());
    assert(world_line_pts_group.size() == camera_lines.size());
    
    // check invert matrix
    for (int i = 0; i<target_covariance_inv.size(); i++) {
        if (target_covariance_inv[i].hasNaN()) {
            printf("Error, numerical error, has NaN.\n");
            return false;
        }
    }
    for (int i = 0; i<world_line_pts_precision.size(); i++) {
        for (int j = 0; j<world_line_pts_precision[i].size(); j++) {
            if (world_line_pts_precision[i][j].hasNaN()) {
                printf("Error, numerical error, has NaN.\n");
                return false;
            }
        }
    }
    
    // count constraint number
    int num_constraint = (int)source_pts.size();
    for (unsigned i = 0; i<world_line_pts_group.size(); i++) {
        num_constraint += (int)world_line_pts_group[i].size();
    }
    if (num_constraint <= 7) {
        return false;
    }
    
    Eigen::Matrix3d R = init_pose.rotation();
    Eigen::Vector3d trs = init_pose.translation(); // translation    
    vnl_point_line_mahalanobis_residual residual(source_pts, target_pts, target_covariance_inv,
                                                 world_line_pts_group, world_line_pts_precision, camera_lines, num_constraint);
    
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
        //lmq.diagnose_outcome();
        residual.getResult(x, refined_pose);
        return true;
    }
    //lmq.diagnose_outcome();
    
    
    return false;
}




