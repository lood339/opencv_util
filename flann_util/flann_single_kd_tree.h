//
//  flann_single_kd_tree.h
//  OLNN
//
//  Created by jimmy on 2016-11-30.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef OLNN_flann_single_kd_tree_h
#define OLNN_flann_single_kd_tree_h

#include <stdio.h>
#include <flann/flann.hpp>
#include <Eigen/Dense>
#include <vector>

using std::vector;

template <class T>
class FlannSingleKDTree
{
    flann::Index<flann::L2<T> > index_;    // store kd tree
    int dim_;
public:
    FlannSingleKDTree(const flann::IndexParams& params = flann::KDTreeSingleIndexParams(10)):index_(params)
    {
        dim_ = 0;
    }
    ~FlannSingleKDTree()
    {
        
    }
    
    void setData(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data, const int leaf_max_size = 10)
    {
        assert(data.rows() > 0);
        dim_ = (int)data.cols();
        
        flann::Matrix<T> dataset(const_cast<T *>(data.data()), (int)data.rows(), dim_);
        index_ = flann::Index< flann::L2<T> >(dataset, flann::KDTreeSingleIndexParams(leaf_max_size));
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
        
        flann::Matrix<T> query_data_wrap(const_cast<T *>(query_data.data()), (int)query_data.rows(), dim_);
        index_.knnSearch(query_data_wrap, indices, dists, knn, flann::SearchParams(num_search_leaf));
    }
};

typedef FlannSingleKDTree<float> FlannSingleKDTree32F;
typedef FlannSingleKDTree<double> FlannSingleKDTree64F;



#endif
