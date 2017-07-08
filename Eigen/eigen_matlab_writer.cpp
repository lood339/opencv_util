//
//  eigen_matlab_writer.cpp
//  OLNN
//
//  Created by jimmy on 2016-12-14.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "eigen_matlab_writer.h"
#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include "vnl_matlab_write.h"

using std::string;

template <class T>
void EigenMatlabWriter::matrix_filewrite(const T & eigen_matrix, const char* file_name,  const char * var_name)
{
    assert(file_name != nullptr);
    assert(var_name != nullptr);
    
    const long row = eigen_matrix.rows();
    const long column = eigen_matrix.cols();
    
    std::fstream os;
    
    
    // const string mat_filename = ((string)file_name + ".mat").c_str();
    os.open(file_name, std::ios::out | std::ios::binary);
    
    double** data_2d = new double*[row];
    
    for(int i = 0; i < row; ++i) {
        data_2d[i] = new double[column];
    }
    
    for(int i=0; i<row; i++)  {
        for(int j=0; j<column; j++) {
            data_2d[i][j]=eigen_matrix(i,j);
        }
    }
    
    vnl_matlab_write<double>(os, data_2d, (unsigned int)row, (unsigned int)column, var_name);
    
    // clean memory
    for (int i = 0; i<row; i++) {
        delete []data_2d[i];
        data_2d[i] = NULL;
    }
    delete []data_2d;
    printf("write to %s\n", file_name);
}

template <class T>
void EigenMatlabWriter::write_vector(const vector<T>& data, const char*file_name, const char * var_name)
{
    Eigen::VectorXd d(data.size());
    for (int i = 0; i<data.size(); i++) {
        d[i] = (double)data[i];
    }
    EigenMatlabWriter::matrix_filewrite<Eigen::VectorXd>(d, file_name, var_name);
}


// matrix
template void EigenMatlabWriter::matrix_filewrite(const Eigen::MatrixXd & eigen_matrix, const char* file_name, const char * var_name);
template void EigenMatlabWriter::matrix_filewrite(const Eigen::MatrixXf & eigen_matrix, const char* file_name, const char * var_name);
template void EigenMatlabWriter::matrix_filewrite(const Eigen::MatrixXi & eigen_matrix, const char* file_name, const char * var_name);

// vector
template void EigenMatlabWriter::matrix_filewrite(const Eigen::VectorXd & eigen_matrix, const char* file_name, const char * var_name);
template void EigenMatlabWriter::matrix_filewrite(const Eigen::VectorXf & eigen_matrix, const char* file_name, const char * var_name);

// vector
template void EigenMatlabWriter::write_vector(const vector<double>& data, const char*file_name, const char * var_name);
template void EigenMatlabWriter::write_vector(const vector<float>& data, const char*file_name, const char * var_name);


