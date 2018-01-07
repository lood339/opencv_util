//
//  UT_ceres.cpp
//  OpenCV_VXL
//
//  Created by jimmy on 4/2/15.
//  Copyright (c) 2015 Nowhere Planet. All rights reserved.
//

#include "UT_ceres.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>

using ceres::Problem;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::CauchyLoss;
using ceres::Solver;
using ceres::Solve;

void test_ceres_tutorial()
{
//    test_ceres_hello_world();
//    test_ceres_analytic_derivative();
//    test_powell_function();
//    test_ceres_data_fit();
    test_ceres_bundle_adjustment();
}

struct CostFunctor{
    template < typename T>
    bool operator()(const T* const x, T*residual) const{
        residual[0] = T(10.0) - x[0];
        return true;
    }
};
void test_ceres_hello_world()
{
    double initial_x = 5.0;
    double x = initial_x;
    
    Problem problem;
    CostFunction * cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
    
    problem.AddResidualBlock(cost_function, NULL, &x);
    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    
    std::cout<<summary.BriefReport()<<std::endl;
    printf("inital x -> x %f %f\n", initial_x, x);
}

class QuadraticCostFunction: public ceres::SizedCostFunction<1, 1>
{
public:
    virtual ~ QuadraticCostFunction(){}
    virtual bool Evaluate(double const* const* parameters,
                          double *residual,
                          double ** jacobians) const
    {
        const double x = parameters[0][0];
        residual[0] = 10 - x;
        if (jacobians != NULL && jacobians[0] != NULL) {
            jacobians[0][0] = -1;
        }
        return true;
    }
    
};


void test_ceres_analytic_derivative()
{
    double initial_x = 0.5;
    double x = initial_x;
    
    Problem problem;
    CostFunction * cost_function = new QuadraticCostFunction();
    
    problem.AddResidualBlock(cost_function, NULL, &x);
    Solver::Options options;
 //   options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    
    std::cout<<summary.BriefReport()<<std::endl;
    printf("inital x -> x %f %f\n", initial_x, x);
}

// functions for Powell's function
struct F1{
    template<typename T>
    bool operator()(const T * const x1, const T* const x2, T*residual) const
    {
        residual[0] = T(x1[0] + T(10.0) * x2[0]);
        return true;
    }
};
struct F2{
    template<typename T>
    bool operator()(const T * const x3, const T* const x4, T*residual) const
    {
        residual[0] = T(sqrt(5.0)) * (x3[0] - x4[0]);
        return true;
    }
};
struct F3{
    template<typename T>
    bool operator()(const T * const x2, const T* const x3, T*residual) const
    {
        residual[0] = T((x2[0] - 2.0*x3[0])*(x2[0] - 2.0*x3[0]));
        return true;
    }
};
struct F4{
    template<typename T>
    bool operator()(const T * const x1, const T* const x4, T*residual) const
    {
        residual[0] = T(sqrt(10.0)) *(x1[0] - x4[0]) * (x1[0] - x4[0]);
        return true;
    }
};

void test_powell_function()
{
    double x1 =  3.0; double x2 = -1.0; double x3 =  0.0; double x4 = 1.0;
    
    Problem problem;
    //
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<F1, 1, 1, 1>(new F1), NULL, &x1, &x2);
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<F2, 1, 1, 1>(new F2), NULL, &x3, &x4);
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<F3, 1, 1, 1>(new F3), NULL, &x2, &x3);
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<F4, 1, 1, 1>(new F4), NULL, &x1, &x4);
    
    Solver::Options options;
    //   options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    
    std::cout<<summary.FullReport()<<std::endl;
  //  printf("inital x -> x %f %f\n", initial_x, x);
}

const int kNumObservations = 67;
const double data[] = {
    0.000000e+00, 1.133898e+00,
    7.500000e-02, 1.334902e+00,
    1.500000e-01, 1.213546e+00,
    2.250000e-01, 1.252016e+00,
    3.000000e-01, 1.392265e+00,
    3.750000e-01, 1.314458e+00,
    4.500000e-01, 1.472541e+00,
    5.250000e-01, 1.536218e+00,
    6.000000e-01, 1.355679e+00,
    6.750000e-01, 1.463566e+00,
    7.500000e-01, 1.490201e+00,
    8.250000e-01, 1.658699e+00,
    9.000000e-01, 1.067574e+00,
    9.750000e-01, 1.464629e+00,
    1.050000e+00, 1.402653e+00,
    1.125000e+00, 1.713141e+00,
    1.200000e+00, 1.527021e+00,
    1.275000e+00, 1.702632e+00,
    1.350000e+00, 1.423899e+00,
    1.425000e+00, 1.543078e+00,
    1.500000e+00, 1.664015e+00,
    1.575000e+00, 1.732484e+00,
    1.650000e+00, 1.543296e+00,
    1.725000e+00, 1.959523e+00,
    1.800000e+00, 1.685132e+00,
    1.875000e+00, 1.951791e+00,
    1.950000e+00, 2.095346e+00,
    2.025000e+00, 2.361460e+00,
    2.100000e+00, 2.169119e+00,
    2.175000e+00, 2.061745e+00,
    2.250000e+00, 2.178641e+00,
    2.325000e+00, 2.104346e+00,
    2.400000e+00, 2.584470e+00,
    2.475000e+00, 1.914158e+00,
    2.550000e+00, 2.368375e+00,
    2.625000e+00, 2.686125e+00,
    2.700000e+00, 2.712395e+00,
    2.775000e+00, 2.499511e+00,
    2.850000e+00, 2.558897e+00,
    2.925000e+00, 2.309154e+00,
    3.000000e+00, 2.869503e+00,
    3.075000e+00, 3.116645e+00,
    3.150000e+00, 3.094907e+00,
    3.225000e+00, 2.471759e+00,
    3.300000e+00, 3.017131e+00,
    3.375000e+00, 3.232381e+00,
    3.450000e+00, 2.944596e+00,
    3.525000e+00, 3.385343e+00,
    3.600000e+00, 3.199826e+00,
    3.675000e+00, 3.423039e+00,
    3.750000e+00, 3.621552e+00,
    3.825000e+00, 3.559255e+00,
    3.900000e+00, 3.530713e+00,
    3.975000e+00, 3.561766e+00,
    4.050000e+00, 3.544574e+00,
    4.125000e+00, 3.867945e+00,
    4.200000e+00, 4.049776e+00,
    4.275000e+00, 3.885601e+00,
    4.350000e+00, 4.110505e+00,
    4.425000e+00, 4.345320e+00,
    4.500000e+00, 4.161241e+00,
    4.575000e+00, 4.363407e+00,
    4.650000e+00, 4.161576e+00,
    4.725000e+00, 4.619728e+00,
    4.800000e+00, 4.737410e+00,
    4.875000e+00, 4.727863e+00,
    4.950000e+00, 4.669206e+00,
};

struct ExponentialResidual{
    ExponentialResidual(double x, double y):
    x_(x), y_(y){}
    
    template<typename T>
    bool operator()(const T* const m, const T* const c, T * residual) const
    {
        residual[0] = T(y_) - exp(m[0] * T(x_) + c[0]);
        return true;
    }
    
    
private:
    const double x_;
    const double y_;
};


void test_ceres_data_fit()
{
    double m = 0.0;
    double c = 0.0;
    
    Problem problem;
    for (int i = 0; i<kNumObservations; i++) {
        CostFunction *cost_function = new AutoDiffCostFunction<ExponentialResidual, 1, 1, 1>(
                                                                                             new ExponentialResidual(data[2*i], data[2*i + 1]));
        problem.AddResidualBlock(cost_function, new CauchyLoss(0.5), &m, &c);
    }
    
    Solver::Options options;
    options.max_num_iterations = 25;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
    std::cout << "Initial m: " << 0.0 << " c: " << 0.0 << "\n";
    std::cout << "Final m: " << m << " c: " << c << "\n";
}

/*
struct SnavelyReprojectionError {
    SnavelyReprojectionError(double obs_x, double obs_y):observed_x(obs_x), observed_y(obs_y){}
    
    template <typename T>
    bool operator()(const T* const camera, const T* const point, T * residual) const
    {
        // camera[0,1,2] are the angle-axis rotation.
        T p[3] = {T(0.0)};
        ceres::AngleAxisRotatePoint(camera, point, p);
        
        // translation
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        
        // computer center of distortion
        T xp = p[0]/p[2];
        T yp = p[1]/p[2];
        
        // apply distortion
        const T & l1 = camera[7];
        const T & l2 = camera[8];
        T r2 = xp * xp + yp * yp;
        T distortion = T(1.0) + r2 * (l1 + l2 * r2);
        
        // projected point position
        const T& focal = camera[6];
        T pre_x = focal * distortion * xp;
        T pre_y = focal * distortion * yp;
        
        // residual
        residual[0] = pre_x - T(observed_x);
        residual[1] = pre_y - T(observed_y);
        
        return true;
    }
    
    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double observed_x,
                                       const double observed_y) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
                                                                                   new SnavelyReprojectionError(observed_x, observed_y)));
    }

    double observed_x;
    double observed_y;
};
 */

// Templated pinhole camera model for used with Ceres. The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
struct SnavelyReprojectionError {
    SnavelyReprojectionError(double observed_x, double observed_y)
    : observed_x(observed_x), observed_y(observed_y) {}
    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T* residuals) const {
        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        T xp = - p[0] / p[2];
        T yp = - p[1] / p[2];
        // Apply second and fourth order radial distortion.
        const T& l1 = camera[7];
        const T& l2 = camera[8];
        T r2 = xp*xp + yp*yp;
        T distortion = T(1.0) + r2 * (l1 + l2 * r2);
        // Compute final projected point position.
        const T& focal = camera[6];
        T predicted_x = focal * distortion * xp;
        T predicted_y = focal * distortion * yp;
        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);
        return true;
    }
    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double observed_x,
                                       const double observed_y) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
                                                                                   new SnavelyReprojectionError(observed_x, observed_y)));
    }
    double observed_x;
    double observed_y;
};


// Read a Bundle Adjustment in the Large dataset.
class BALProblem {
public:
    ~BALProblem() {
        delete[] point_index_;
        delete[] camera_index_;
        delete[] observations_;
        delete[] parameters_;
    }
    int num_observations() const { return num_observations_; }
    const double* observations() const { return observations_; }
    double* mutable_cameras() { return parameters_; }
    double* mutable_points() { return parameters_ + 9 * num_cameras_; }
    double* mutable_camera_for_observation(int i) {
        return mutable_cameras() + camera_index_[i] * 9;
    }
    double* mutable_point_for_observation(int i) {
        return mutable_points() + point_index_[i] * 3;
    }
    bool LoadFile(const char* filename) {
        FILE* fptr = fopen(filename, "r");
        if (fptr == NULL) {
            return false;
        };
        FscanfOrDie(fptr, "%d", &num_cameras_);
        FscanfOrDie(fptr, "%d", &num_points_);
        FscanfOrDie(fptr, "%d", &num_observations_);
        point_index_ = new int[num_observations_];
        camera_index_ = new int[num_observations_];
        observations_ = new double[2 * num_observations_];
        num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
        parameters_ = new double[num_parameters_];
        for (int i = 0; i < num_observations_; ++i) {
            FscanfOrDie(fptr, "%d", camera_index_ + i);
            FscanfOrDie(fptr, "%d", point_index_ + i);
            for (int j = 0; j < 2; ++j) {
                FscanfOrDie(fptr, "%lf", observations_ + 2*i + j);
            }
        }
        for (int i = 0; i < num_parameters_; ++i) {
            FscanfOrDie(fptr, "%lf", parameters_ + i);
        }
        return true;
    }
private:
    template<typename T>
    void FscanfOrDie(FILE *fptr, const char *format, T *value) {
        int num_scanned = fscanf(fptr, format, value);
        if (num_scanned != 1) {
            LOG(FATAL) << "Invalid UW data file.";
        }
    }
    int num_cameras_;
    int num_points_;
    int num_observations_;
    int num_parameters_;
    int* point_index_;
    int* camera_index_;
    double* observations_;
    double* parameters_;
};

void test_ceres_bundle_adjustment()
{
    BALProblem bal_problem;
    bool isRead = bal_problem.LoadFile("/Users/jimmy/Desktop/bundle_adjustment_data/problem-49-7776-pre.txt");
    assert(isRead);
    
    
    const double* observations = bal_problem.observations();
    // Create residuals for each observation in the bundle adjustment problem. The
    // parameters for cameras and points are added automatically.
    ceres::Problem problem;
    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        // Each Residual block takes a point and a camera as input and outputs a 2
        // dimensional residual. Internally, the cost function stores the observed
        // image location and compares the reprojection against the observation.
        ceres::CostFunction* cost_function =
        SnavelyReprojectionError::Create(observations[2 * i + 0],
                                         observations[2 * i + 1]);
        problem.AddResidualBlock(cost_function,
                                 NULL /* squared loss */,
                                 bal_problem.mutable_camera_for_observation(i),
                                 bal_problem.mutable_point_for_observation(i));
    }
    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    
    
}































































