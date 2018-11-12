//
//  main.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-07-26.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#if 1


#include "opt_ptz_camera.h"
#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using cvx::perspective_camera;
using cvx::calibration_matrix;

/*
 bool estimateCommomCameraCenterAndRotation(const vector<vector<Vector3d>> & wld_pts,
 const vector<vector<Vector2d>> & img_pts,
 const vector<perspective_camera> & init_cameras,
 const Vector3d & init_common_rotation,
 Vector3d & estimated_camera_center,
 Vector3d & estimated_common_rotation,
 vector<perspective_camera > & estimated_cameras);
 */
extern "C" {
    void estimateCommomCameraCenterAndRotation(const double* model_pts,
                                               const int rows,
                                               const int cols,
                                               const double* input_init_cameras,
                                               const int camera_num,
                                               const int camera_param_len,
                                               const double* input_init_common_rotation,
                                               double* opt_cameras,
                                               double* commom_center,
                                               double* commom_rotation)
    
    
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
        Eigen::AlignedBox<double, 2> im_size(im_w, im_h);
        
        vector<vector<Vector3d>> wld_pts;
        vector<vector<Vector2d>> img_pts;
        for(int i = 0; i<init_cameras.size(); i++) {
            vector<Vector3d> cur_wld_pts;
            vector<Vector2d> cur_img_pts;
            for (int r = 0; r<wld_mat.rows(); r++) {
                Eigen::Vector3d p = wld_mat.row(r);
                Eigen::Vector2d q = init_cameras[i].project3d(p);
                if (im_size.contains(q)) { // inside image
                    cur_wld_pts.push_back(p);
                    cur_img_pts.push_back(q);
                }
            }
            assert(cur_wld_pts.size() >= 2);
            
            wld_pts.push_back(cur_wld_pts);
            img_pts.push_back(cur_img_pts);
        }
        
        Vector3d estimated_camera_center;
        Vector3d estimated_common_rotation;
        vector<perspective_camera > estimated_cameras;
        
        bool is_estimated = cvx::estimateCommomCameraCenterAndRotation(wld_pts,
                                                   img_pts,
                                                   init_cameras,
                                                   init_rod,
                                                   estimated_camera_center,
                                                   estimated_common_rotation,
                                                   estimated_cameras);
        if (!is_estimated) {
            printf("Warning: estimate PTZ camera center and tripod rotation failed!");
            return;
        }
        
        // step 3: output
        
        /*
        double* opt_cameras,
        double* commom_center,
        double* commom_rotation
         */
       
    }
}

#endif

