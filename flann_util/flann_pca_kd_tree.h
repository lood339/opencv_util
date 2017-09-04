//
//  flann_pca_kd_tree.h
//  OLNN
//
//  Created by jimmy on 2016-12-14.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef OLNN_flann_pca_kd_tree_h
#define OLNN_flann_pca_kd_tree_h

// KD-tree using PCA
#include <stdio.h>
#include <flann/flann.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include "RedSVD.h"
#include "yael_io.h"

using std::vector;

enum flann_pca_t
{
    FLANN_PCA_ORIGIN_DIMENSION = 0,  /* keep original dimension */
    FLANN_PCA_EXPLAINED_VARIANCE = 1,
    FLANN_PCA_USER_DEFINED_DIMENSION = 2
};


template <class T>
class FlannPCAKDTree
{
    using row_major_matrix_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    
    flann::Index<flann::L2<T> > index_;    // store kd tree
    
    int dim_;
    int pca_dim_;
    row_major_matrix_type cp_;  // component V in USV^T
    row_major_matrix_type pca_base_data_;  // save PCA version of basedata
public:
    FlannPCAKDTree(const flann::IndexParams& params = flann::KDTreeIndexParams(4)):index_(params)
    {
        dim_ = 0;
        pca_dim_ = 0;
    }
    
    ~FlannPCAKDTree()
    {
        
    }
    
    void setDataPCADim(const row_major_matrix_type & learn_data, const int pca_dim)
    {
        assert(learn_data.cols() >= pca_dim);
        pca_dim_ = pca_dim;
        
        RedSVD::RedPCA<row_major_matrix_type> pca(learn_data, pca_dim_);
        cp_ = pca.components();
     //   printf("PCA component rows, cols are %ld %ld\n", cp_.rows(), cp_.cols());
    }
    
    // explained_var: negative value for all the components
    void setDataExplainedVariance(const row_major_matrix_type & learn_data, const double explained_var = 0.9)
    {
        // set PCA dim from explained_var
        assert(explained_var <= 1.0);
        const long rows = learn_data.rows();
        const long cols = learn_data.cols();
        
        RedSVD::RedPCA<row_major_matrix_type> pca(learn_data, cols);
        Eigen::Matrix<T, Eigen::Dynamic, 1> eigen_values = pca.singularValues();
        pca_dim_ = (int)cols;
        if (explained_var < 0.0) {
            pca_dim_ = (int)cols;
        }
        else {
            double sum_eigen = eigen_values.array().sum();
            double accu_eigen = 0.0;
            
            for (int i = 0; i<cols; i++) {
                accu_eigen += eigen_values[i];
                if ((accu_eigen/sum_eigen) >= explained_var) {
                    pca_dim_ = i + 1;
                    break;
                }
            }
        }
        
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> V = pca.components();
        cp_ = V.middleCols(0, pca_dim_);
    }
    
    void setDataExplainedVariance(const row_major_matrix_type & learn_data, const double sample_ratio,
                                  const double explained_var = 0.9)
    {
        // set PCA dim from explained_var
        assert(explained_var <= 1.0);
        assert(sample_ratio <= 1.0);
        const long rows = learn_data.rows();
        const long cols = learn_data.cols();
        
        const long sampled_num = rows * sample_ratio;
        row_major_matrix_type sample_data = row_major_matrix_type::Zero(sampled_num, cols);
        for(int i = 0; i < sampled_num; ++i) {
            sample_data.row(i) = learn_data.row(rand()%cols);
        }
        
        RedSVD::RedPCA<row_major_matrix_type> pca(sample_data, cols);
        Eigen::Matrix<T, Eigen::Dynamic, 1> eigen_values = pca.singularValues();
        pca_dim_ = (int)cols;
        if (explained_var < 0.0) {
            pca_dim_ = (int)cols;
        }
        else {
            double sum_eigen = eigen_values.array().sum();
            double accu_eigen = 0.0;
            
            for (int i = 0; i<cols; i++) {
                accu_eigen += eigen_values[i];
                if ((accu_eigen/sum_eigen) >= explained_var) {
                    pca_dim_ = i + 1;
                    break;
                }
            }
        }
        
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> V = pca.components();
        cp_ = V.middleCols(0, pca_dim_);
        printf("PCA dimension is %d\n", pca_dim_);
    }

    
    void setData(const row_major_matrix_type & base_data, const int trees = 4)
    {
        assert(base_data.rows() > 0);
        assert(pca_dim_ > 0);
        
        dim_ = (int)base_data.cols();
        assert(pca_dim_ <= dim_);
        
        pca_base_data_ = base_data * cp_;
        
        flann::Matrix<T> dataset(const_cast<T *>(pca_base_data_.data()), pca_base_data_.rows(), pca_base_data_.cols());
        index_ = flann::Index< flann::L2<T> >(dataset, flann::KDTreeIndexParams(trees));
        index_.buildIndex();
    }
    
    void search(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & query_data,
                vector<vector<int> > & indices,
                vector<vector<T> > & dists,
                const int knn,
                int num_search_leaf = 128) const
    {
        assert(query_data.rows() > 0);
        assert(dim_ == query_data.cols());
        
        row_major_matrix_type pca_query_data = query_data * cp_;        
      
        flann::Matrix<T> query_data_wrap(const_cast<T *>(pca_query_data.data()), pca_query_data.rows(), pca_query_data.cols());
        index_.knnSearch(query_data_wrap, indices, dists, knn, flann::SearchParams(num_search_leaf));
    }
    
    void loadIndex(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data,
                   const char *file_name)
    {
        
        // load PCA component
        std::string cp_file(file_name);
        cp_file += std::string("_pca_cp.fvecs");
        bool is_read = YaelIO::read_fvecs_file(cp_file.c_str(), cp_);
        assert(is_read);
        
        // load index
        dim_ = data.cols();
        flann::SavedIndexParams save_param(file_name);
        pca_base_data_ = data * cp_;
        pca_dim_ = pca_base_data_.cols();
        flann::Matrix<T> dataset(const_cast<T *>(pca_base_data_.data()), (int)pca_base_data_.rows(), pca_base_data_.cols());
        index_ = flann::Index< flann::L2<T> >(dataset, save_param);
    }
    
    void saveIndex(const char *file_name)
    {
        // save index
        index_.save(std::string(file_name));
        
        // save PCA component
        assert(sizeof(cp_(0, 0)) == sizeof(float));
        
        std::string cp_file(file_name);
        cp_file += std::string("_pca_cp.fvecs");
        bool is_write = YaelIO::write_fvecs_file(cp_file.c_str(), cp_);
        assert(is_write);
        printf("save to files: %s\n %s\n\n", file_name, cp_file.c_str());
    }

};

typedef FlannPCAKDTree<float>  FlannPCAKDTree32F;
typedef FlannPCAKDTree<double> FlannPCAKDTree3264F;

#endif
