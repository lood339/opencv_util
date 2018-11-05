//
//  eigen_least_square.cpp
//  Classifer_RF
//
//  Created by jimmy on 2017-03-13.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "eigen_least_square.h"
#include <assert.h>
#include <Eigen/Dense>


using Eigen::MatrixXd;
using Eigen::VectorXd;

bool EigenLeastSquare::solver(vector<map<int, double> > & leftVec, vector<double> & rightVec,
                            bool overConstraint, int var_Num, double *result)
{
    assert(leftVec.size() == rightVec.size());
    assert(leftVec.size() >= var_Num);
    assert(overConstraint);
    
    // construct A from leftVec
    // construct b from rightVec
    MatrixXd A = MatrixXd::Zero(leftVec.size(), var_Num);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(rightVec.size(), 1);
    for (int i = 0; i<leftVec.size(); i++) {
        for (map<int, double>::iterator it = leftVec[i].begin(); it != leftVec[i].end(); it++) {
            int col = it->first;
            double val = it->second;
            A(i, col) = val;
        }
        b[i] = rightVec[i];
    }
    // ret: the solution of Ax = b
    Eigen::VectorXd ret = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    
    assert(ret.size() == var_Num);
    for (int i = 0; i<var_Num; i++) {
        result[i] = ret[i];
    }
    return true;
}
