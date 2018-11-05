//
//  eigen_matlab_writer.h
//  OLNN
//
//  Created by jimmy on 2016-12-14.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __OLNN__eigen_matlab_writer__
#define __OLNN__eigen_matlab_writer__

#include <stdio.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

// write Eigen matrix, vector to .mat file
// read/write is slow and only suitable for small matrix
// only test on Mac OS

using std::vector;

class EigenMatlabWriter
{
public:
    template<class T>  // file_name: save to .mat,
    static void matrix_filewrite(const T & eigen_matrix, const char* file_name, const char * var_name = "data");
    
    template<class T>  // float, double
    static void write_vector(const vector<T>& data, const char*file_name, const char * var_name = "data");
};


#endif /* defined(__OLNN__eigen_matlab_writer__) */
