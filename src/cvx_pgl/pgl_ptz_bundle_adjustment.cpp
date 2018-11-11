//
//  pgl_ptz_boundle_adjustment.cpp
//  CalibMeMatching
//
//  Created by jimmy on 2017-08-05.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "pgl_ptz_bundle_adjustment.h"
#include <iostream>

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

using Eigen::VectorXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using std::cout;
using std::endl;

namespace cvx {
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
                           vector<Vector3d> & pan_tilt_fl) const
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
    
    bool bundle_adjustment(ptz_camera& camera1, ptz_camera& camera2,
                            MatrixXd& location1,  MatrixXd& location2)
    {
        //printf("bundle_adjustment only used for debug.\n");
        //assert(0);
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
        cout<<"ptz1 before: "<<ptz1.transpose()<<" after: "<<pan_tilt_fl[0].transpose()<<endl;
        cout<<"ptz2 before: "<<ptz2.transpose()<<" after: "<<pan_tilt_fl[1].transpose()<<endl;
        
        return true;
    }
    
    // return -1: not found, otherwise index in database
    static int findNearestNeighbor(const Eigen::Vector2d& p,
                                   const vector<Eigen::Vector2d> & database,
                                   double threshold)
    {
        int index = -1;
        double min_dist = INT_MAX;
        for (int i = 0; i<database.size(); i++) {
            double dist = (p - database[i]).norm();
            if (dist < threshold && dist < min_dist) {
                min_dist = dist;
                index = i;
            }
        }
        return index;
    }
    static vector<int> histgram(const vector<int> & data)
    {
        int min_v = *std::min_element(data.begin(), data.end());
        int max_v = *std::max_element(data.begin(), data.end());
        assert(min_v >= 0);
        assert(max_v >= 0);
        
        vector<int> h(max_v+1, 0);
        for (int i = 0; i<data.size(); i++) {
            h[data[i]]++;
        }
        return h;
    }   
    
    bool keypoint_associateion(const vector<std::pair<int, int> > & image_ids,
                               const vector<std::pair<ptz_camera, ptz_camera> > & cameras,
                               const vector<std::pair<MatrixXd, MatrixXd> > & keypoints)
    {
        printf("only for debug purpose\n");
        assert(0);
        assert(image_ids.size() == cameras.size());
        assert(image_ids.size() == keypoints.size());
        
        // image number
        int image_num = 0;
        for (int i = 0; i<image_ids.size(); i++) {
            image_num = std::max(image_num, image_ids[i].first);
            image_num = std::max(image_num, image_ids[i].second);
        }
        image_num++;
        assert(image_num >= 2);
        
        Eigen::MatrixXi adjacency = Eigen::MatrixXi::Zero(image_num, image_num);
        vector<Eigen::Vector2d> spherical_points;       // like 3D model
        vector<int>             point_count;            // number of pixel location for each 3D point
        vector<vector<int> > visivility(image_num);  // like 3D point index of each image location
        vector<vector<Eigen::Vector2d> > all_image_points(image_num); // 2D image points emerged from multiple images
        const double same_point_dist_threshold = 1.0;
        
        // loop each image pairs
        for (int i = 0; i<image_ids.size(); i++) {
            int im_id1 = image_ids[i].first;
            int im_id2 = image_ids[i].second;
            assert(im_id1 != im_id2);
            
            bool has_visit_image1 = (adjacency(im_id1, im_id1) != 0);
            bool has_visit_image2 = (adjacency(im_id2, im_id2) != 0);
            
            MatrixXd keypoints1 = keypoints[i].first;
            MatrixXd keypoints2 = keypoints[i].second;
            assert(keypoints1.rows() == keypoints2.rows());
            assert(keypoints1.cols() == 2);
            
            ptz_camera ptz1 = cameras[i].first;
            ptz_camera ptz2 = cameras[i].second;
            
            // plance image id in increasing order
            if (im_id1 > im_id2) {
                std::swap(im_id1, im_id2);
                std::swap(has_visit_image1, has_visit_image2);
                std::swap(keypoints1, keypoints2);
                ptz1 = cameras[i].second;
                ptz2 = cameras[i].first;
            }
            
            
            if (!has_visit_image1 && !has_visit_image2) {
                // first pair of matching
                for (int r = 0; r<keypoints1.rows(); r++) {
                    Eigen::Vector2d p1 = keypoints1.row(r);
                    Eigen::Vector2d p2 = keypoints2.row(r);
                    Eigen::Vector2d avg_pt = 0.5 * (ptz1.back_project(p1.x(), p1.y()) + ptz2.back_project(p2.x(), p2.y()));
                    int cur_s_index = (int)spherical_points.size();
                    
                    // update model and index in each image
                    visivility[im_id1].push_back(cur_s_index);
                    visivility[im_id2].push_back(cur_s_index);
                    spherical_points.push_back(avg_pt);
                    point_count.push_back(2);
                    
                    all_image_points[im_id1].push_back(p1);
                    all_image_points[im_id2].push_back(p2);
                    
                    assert(point_count.size() == spherical_points.size());
                }
            }
            else if(has_visit_image1 && !has_visit_image2) {
                // only exit image 1 in graph
                vector<int> cur_point_index(keypoints1.rows(), -1);
                for (int r = 0; r<keypoints1.rows(); r++) {
                    Eigen::Vector2d p = keypoints1.row(r);
                    int nn_index = findNearestNeighbor(p, all_image_points[im_id1], same_point_dist_threshold);
                    cur_point_index[r] = nn_index;
                }
                
                // add image point to the 3D model
                for (int j = 0; j<cur_point_index.size(); j++) {
                    int index = cur_point_index[j];
                    Eigen::Vector2d p1 = keypoints1.row(j);
                    Eigen::Vector2d p2 = keypoints2.row(j);
                    Eigen::Vector2d avg_pt = 0.5 * (ptz1.back_project(p1.x(), p1.y()) + ptz2.back_project(p2.x(), p2.y()));
                    
                    if (index == -1) {
                        // new point
                        int cur_s_index = (int)spherical_points.size();
                        
                        // update model and index in each image
                        visivility[im_id1].push_back(cur_s_index);
                        visivility[im_id2].push_back(cur_s_index);
                        spherical_points.push_back(avg_pt);
                        point_count.push_back(2);
                        
                        all_image_points[im_id1].push_back(p1);
                        all_image_points[im_id2].push_back(p2);
                        
                        assert(point_count.size() == spherical_points.size());
                        
                    }
                    else {
                        // associate point with the point in the 3D model
                        int s_index = visivility[im_id1][index];
                        assert(s_index >= 0 && s_index < spherical_points.size());
                        avg_pt = (2*avg_pt + spherical_points[s_index] * point_count[s_index])/(point_count[s_index] + 2);
                        
                        spherical_points[s_index] = avg_pt;
                        point_count[s_index] += 2;
                        
                        visivility[im_id2].push_back(s_index);
                        all_image_points[im_id2].push_back(p2);
                    }
                }
            }
            else if (has_visit_image1 && has_visit_image2) {
                // both image 1 and image 2 are seen before
                vector<int> image1_point_index(keypoints1.rows(), -1);
                for (int r = 0; r<keypoints1.rows(); r++) {
                    Eigen::Vector2d p = keypoints1.row(r);
                    int nn_index = findNearestNeighbor(p, all_image_points[im_id1], same_point_dist_threshold);
                    image1_point_index[r] = nn_index;
                }
                vector<int> image2_point_index(keypoints2.rows(), -1);
                for (int r = 0; r<keypoints2.rows(); r++) {
                    Eigen::Vector2d p = keypoints2.row(r);
                    int nn_index = findNearestNeighbor(p, all_image_points[im_id2], same_point_dist_threshold);
                    image2_point_index[r] = nn_index;
                }
                assert(image1_point_index.size() == image2_point_index.size());
                
                // update 3D model
                for (int j = 0; j<image1_point_index.size(); j++) {
                    int index1 = image1_point_index[j];
                    int index2 = image2_point_index[j];
                    //printf("index1 index2 %d %d\n", index1, index2);
                    if (index1 == -1 || index2 == -1) {
                        // new point
                        Eigen::Vector2d p1 = keypoints1.row(j);
                        Eigen::Vector2d p2 = keypoints2.row(j);
                        Eigen::Vector2d avg_pt = 0.5 * (ptz1.back_project(p1.x(), p1.y()) + ptz2.back_project(p2.x(), p2.y()));
                        int cur_s_index = (int)spherical_points.size();
                        
                        // update model and index in each image
                        visivility[im_id1].push_back(cur_s_index);
                        visivility[im_id2].push_back(cur_s_index);
                        spherical_points.push_back(avg_pt);
                        point_count.push_back(2);
                        
                        all_image_points[im_id1].push_back(p1);
                        all_image_points[im_id2].push_back(p2);
                        
                        assert(point_count.size() == spherical_points.size());
                    }
                    else {
                        int s_index1 = visivility[im_id1][index1];
                        int s_index2 = visivility[im_id2][index2];
                        
                        //printf("s_index1, s_index2 %d %d\n", s_index1, s_index2);
                        
                        if (s_index1 == s_index2) {
                            assert(s_index1 >= 0 && s_index1 < spherical_points.size());
                            Eigen::Vector2d p1 = keypoints1.row(j);
                            Eigen::Vector2d p2 = keypoints2.row(j);
                            Eigen::Vector2d avg_pt = 0.5 * (ptz1.back_project(p1.x(), p1.y()) + ptz2.back_project(p2.x(), p2.y()));
                            avg_pt = (avg_pt*2 + spherical_points[s_index1] * point_count[s_index1])/(point_count[s_index1] + 2); // average of all observation
                            
                            spherical_points[s_index1] = avg_pt;
                            point_count[s_index1] += 2;
                        }
                        else if (s_index1 != s_index2) {
                            //printf("Warning: data assiociation conflict! Treat as new data point\n");
                            Eigen::Vector2d p1 = keypoints1.row(j);
                            Eigen::Vector2d p2 = keypoints2.row(j);
                            Eigen::Vector2d avg_pt = 0.5 * (ptz1.back_project(p1.x(), p1.y()) + ptz2.back_project(p2.x(), p2.y()));
                            int cur_s_index = (int)spherical_points.size();
                            
                            // update model and index in each image
                            visivility[im_id1].push_back(cur_s_index);
                            visivility[im_id2].push_back(cur_s_index);
                            spherical_points.push_back(avg_pt);
                            point_count.push_back(2);
                            
                            all_image_points[im_id1].push_back(p1);
                            all_image_points[im_id2].push_back(p2);
                            
                            assert(point_count.size() == spherical_points.size());
                        }
                    }
                }
            }
            else {
                printf("Error: image id must be in order. \n");
                assert(0);
            }
            
            assert(point_count.size() == spherical_points.size());            
            
            // update adjacency map
            adjacency(im_id1, im_id1) = 1;
            adjacency(im_id2, im_id2) = 1;
            adjacency(im_id1, im_id2) = 1;
            adjacency(im_id2, im_id1) = 1;
           
            vector<int> hist = histgram(point_count);
            printf("sphercial point number %lu\n", spherical_points.size());
            for (int j = 0; j<hist.size(); j++) {
                printf("%d: %d\n", j, hist[j]);
            }
            for (int j = 0; j<visivility.size(); j++) {
                assert(visivility[j].size() == all_image_points[j].size());
            }
            
            
        }
        
        
        
        
        return true;
    }
}