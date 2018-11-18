//
//  main.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-07-26.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#if 1

#include "cvx_opt_python.h"

#include "opt_ptz_camera.h"
#include "opt_broadcast_camera.h"
#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using cvx::perspective_camera;
using cvx::calibration_matrix;

extern "C" {
    void estimateCommonCameraCenterAndRotation(const double* model_pts,
                                               const int rows,
                                               const int cols,
                                               const double* input_init_cameras,
                                               const int camera_num,
                                               const int camera_param_len,
                                               const double* input_init_common_rotation,
                                               double* output_cameras,
                                               double* output_ptzs,
                                               double* output_commom_center,
                                               double* output_commom_rotation)
    
    
    {
        assert(rows >= 2);
        assert(cols == 3);
        assert(camera_param_len == 9); // ppx, ppy, fl, rx, ry, rz, cx, cy, cz
        
        // step 1: input data
        
        // model points
        Eigen::MatrixXd wld_mat(rows, cols);
        for (int r = 0; r < rows; r++) {
            for(int c = 0; c<cols; c++) {
                wld_mat(r, c) = model_pts[r * cols + c];
            }
        }
        
        // camera parameters
        vector<perspective_camera> init_cameras;
        for(int i = 0; i<camera_num; i++) {
            const double *p = input_init_cameras + i * camera_param_len;
            Vector2d pp(p[0], p[1]);
            Vector3d rod(p[3], p[4], p[5]);
            Vector3d cc(p[6], p[7], p[8]);
            double fl = p[2];
            
            perspective_camera camera;
            calibration_matrix K(fl, pp);
            camera.set_calibration(K);
            camera.set_camera_center(cc);
            camera.set_rotation(rod);
            init_cameras.push_back(camera);
        }

        // @todo how to automatically get this rotation?
        Eigen::Vector3d init_rod(input_init_common_rotation[0],
                                 input_init_common_rotation[1],
                                 input_init_common_rotation[2]);
                                 

        // step 2: optimize PTZ parameters
        const int im_w = init_cameras[0].get_calibration().principal_point().x() * 2;
        const int im_h = init_cameras[0].get_calibration().principal_point().y() * 2;
        assert(im_w > 0 && im_h > 0);

        Eigen::AlignedBox<double, 2> im_size(Vector2d(0, 0), Vector2d(im_w, im_h));
        vector<vector<Vector3d>> wld_pts;
        vector<vector<Vector2d>> img_pts;
        for(int i = 0; i<init_cameras.size(); i++) {
            vector<Vector3d> cur_wld_pts;
            vector<Vector2d> cur_img_pts;
            perspective_camera camera = init_cameras[i];
            for (int r = 0; r<wld_mat.rows(); r++) {
                Eigen::Vector3d p = wld_mat.row(r);
                Eigen::Vector2d q = camera.project3d(p);
                if (im_size.contains(q)) { // inside image
                    cur_wld_pts.push_back(p);
                    cur_img_pts.push_back(q);                    
                }
            }
            //std::cout<<std::endl;
            assert(cur_wld_pts.size() >= 2);
            //printf("%lu\n", cur_img_pts.size());
            
            wld_pts.push_back(cur_wld_pts);
            img_pts.push_back(cur_img_pts);
        }
        assert(wld_pts.size() == img_pts.size());
        
        Vector3d estimated_camera_center;
        Vector3d estimated_common_rotation;
        vector<perspective_camera > estimated_cameras;
        vector<Eigen::Vector3d> estimated_ptzs;
        bool is_estimated = cvx::estimateCommomCameraCenterAndRotation(wld_pts,
                                                                       img_pts,
                                                                       init_cameras,
                                                                       init_rod,
                                                                       estimated_camera_center,
                                                                       estimated_common_rotation,
                                                                       estimated_cameras,
                                                                       estimated_ptzs);
         //printf("5");

        if (!is_estimated) {
            printf("Warning: estimate PTZ camera center and tripod rotation failed!\n");
            //for (int i = 0; i<estimated_ptzs.size(); i++) {
            //    std::cout<<estimated_ptzs[i].transpose()<<std::endl;
            //}
        }
        
        // step 3: output
        for (int i = 0; i<camera_num; i++) {
            cvx::perspective_camera camera = estimated_cameras[i];
            output_cameras[i * 9 + 0] = camera.get_calibration().principal_point().x();
            output_cameras[i * 9 + 1] = camera.get_calibration().principal_point().y();
            output_cameras[i * 9 + 2] = camera.get_calibration().focal_length();
            
            output_cameras[i * 9 + 3] = camera.get_rotation().as_rodrigues().x();
            output_cameras[i * 9 + 4] = camera.get_rotation().as_rodrigues().y();
            output_cameras[i * 9 + 5] = camera.get_rotation().as_rodrigues().z();
            
            output_cameras[i * 9 + 6] = estimated_camera_center.x();
            output_cameras[i * 9 + 7] = estimated_camera_center.y();
            output_cameras[i * 9 + 8] = estimated_camera_center.z();
            
            
            output_ptzs[i * 3 + 0] = estimated_ptzs[i].x();
            output_ptzs[i * 3 + 1] = estimated_ptzs[i].y();
            output_ptzs[i * 3 + 2] = estimated_ptzs[i].z();            
        }
        
        output_commom_center[0] = estimated_camera_center.x();
        output_commom_center[1] = estimated_camera_center.y();
        output_commom_center[2] = estimated_camera_center.z();
        
        output_commom_rotation[0] = estimated_common_rotation.x();
        output_commom_rotation[1] = estimated_common_rotation.y();
        output_commom_rotation[2] = estimated_common_rotation.z();        
    }
    
    void estimateCommonCameraCenterAndRotationAndDisplacment(const double* model_3d_points,
                                                             const int point_num,
                                                             const double* input_init_cameras,
                                                             const int camera_num,
                                                             const double* init_common_rotation,
                                                             double* output_cameras,
                                                             double* output_ptzs,
                                                             double* output_shared_parameters)
    {
        assert(point_num >= 2);
        
        // step 1: input data
        
        const int point_dim = 3;
        const int camera_param_dim = 9;
        // model points
        Eigen::MatrixXd wld_mat(point_num, point_dim);
        for (int r = 0; r < point_num; r++) {
            for(int c = 0; c<point_dim; c++) {
                wld_mat(r, c) = model_3d_points[r * point_dim + c];
            }
        }
        
        // camera parameters
        vector<perspective_camera> init_cameras;
        for(int i = 0; i<camera_num; i++) {
            const double *p = input_init_cameras + i * camera_param_dim;
            Vector2d pp(p[0], p[1]);
            Vector3d rod(p[3], p[4], p[5]);
            Vector3d cc(p[6], p[7], p[8]);
            double fl = p[2];
            
            perspective_camera camera;
            calibration_matrix K(fl, pp);
            camera.set_calibration(K);
            camera.set_camera_center(cc);
            camera.set_rotation(rod);
            init_cameras.push_back(camera);
        }
        
        // @todo how to automatically get this rotation?
        Eigen::Vector3d init_rod(init_common_rotation[0],
                                 init_common_rotation[1],
                                 init_common_rotation[2]);
        
        
        // step 2: optimize PTZ parameters
        const int im_w = init_cameras[0].get_calibration().principal_point().x() * 2;
        const int im_h = init_cameras[0].get_calibration().principal_point().y() * 2;
        assert(im_w > 0 && im_h > 0);
        printf("image width, height %d %d\n", im_w, im_h);
        
        Eigen::AlignedBox<double, 2> im_size(Vector2d(0, 0), Vector2d(im_w, im_h));
        vector<vector<Vector3d>> wld_pts;
        vector<vector<Vector2d>> img_pts;
        for(int i = 0; i<init_cameras.size(); i++) {
            vector<Vector3d> cur_wld_pts;
            vector<Vector2d> cur_img_pts;
            perspective_camera camera = init_cameras[i];
            for (int r = 0; r<wld_mat.rows(); r++) {
                Eigen::Vector3d p = wld_mat.row(r);
                Eigen::Vector2d q = camera.project3d(p);
                if (im_size.contains(q)) { // inside image
                    cur_wld_pts.push_back(p);
                    cur_img_pts.push_back(q);
                }
            }
            assert(cur_wld_pts.size() >= 2);            
            wld_pts.push_back(cur_wld_pts);
            img_pts.push_back(cur_img_pts);
        }
        assert(wld_pts.size() == img_pts.size());
        
        
        vector<broadcast_camera> estimated_cameras;        
        bool is_estimated = cvx::estimateBroadcastCameras(wld_pts,
                                                          img_pts,
                                                          init_cameras,
                                                          init_rod,
                                                          estimated_cameras);
        if (!is_estimated) {
            printf("Warning: broadcast camera estimation has larger reprojecion errors\n");
        }
        
        // step 3: output
        for (int i = 0; i<camera_num; i++) {
            cvx::broadcast_camera camera = estimated_cameras[i];
            output_cameras[i * 9 + 0] = camera.get_calibration().principal_point().x();
            output_cameras[i * 9 + 1] = camera.get_calibration().principal_point().y();
            output_cameras[i * 9 + 2] = camera.get_calibration().focal_length();
            
            output_cameras[i * 9 + 3] = camera.get_rotation().as_rodrigues().x();
            output_cameras[i * 9 + 4] = camera.get_rotation().as_rodrigues().y();
            output_cameras[i * 9 + 5] = camera.get_rotation().as_rodrigues().z();
            
            output_cameras[i * 9 + 6] = camera.get_camera_center().x();
            output_cameras[i * 9 + 7] = camera.get_camera_center().y();
            output_cameras[i * 9 + 8] = camera.get_camera_center().z();
            
            output_ptzs[i * 3 + 0] = camera.pan();
            output_ptzs[i * 3 + 1] = camera.tilt();
            output_ptzs[i * 3 + 2] = camera.focal_length();
        }
        
        Eigen::Vector3d cc = estimated_cameras[0].get_camera_center();
        Eigen::Vector3d rod = estimated_cameras[0].get_shared_rotation();
        Eigen::VectorXd lambda = estimated_cameras[0].lambda();
        assert(lambda.size() == 6);
        
        output_shared_parameters[0] = cc.x();
        output_shared_parameters[1] = cc.y();
        output_shared_parameters[2] = cc.z();
        
        output_shared_parameters[3] = rod.x();
        output_shared_parameters[4] = rod.y();
        output_shared_parameters[5] = rod.z();
        
        for(int i = 0; i<6; i++) {
            output_shared_parameters[6 + i] = lambda[i];
        }
    }
    
    void broadcastCameraProjection(const double* camera_parameters,
                                   const double* model_3d_points,
                                   const int point_num,
                                   double* image_points)
    {
        // 1. input
        Eigen::Vector3d cc(camera_parameters[0], camera_parameters[1], camera_parameters[2]);
        Eigen::Vector3d base_rod(camera_parameters[3], camera_parameters[4], camera_parameters[5]);
        Eigen::VectorXd lambda(6);
        for (int i = 0; i<6; i++) {
            lambda[i] = camera_parameters[6 + i];
        }
        const int shared_dim = 12;
        Eigen::Vector2d pp(camera_parameters[shared_dim + 0], camera_parameters[shared_dim + 1]);
        double pan = camera_parameters[shared_dim + 2];
        double tilt = camera_parameters[shared_dim + 3];
        double fl = camera_parameters[shared_dim + 4];
        
        // 2. project and output
        broadcast_camera camera(pp, cc, base_rod, lambda, pan, tilt, fl);
        for (int i = 0; i<point_num; i++) {
            Eigen::Vector3d p(model_3d_points + i*3);
            Eigen::Vector2d q = camera.project3d(p);
            image_points[i * 2 + 0] = q.x();
            image_points[i * 2 + 1] = q.y();
        }
    }
}




#endif

