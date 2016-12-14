//
//  eigenFlann.h
//  RGBD_RF
//
//  Created by jimmy on 2016-09-06.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __RGBD_RF__eigenFlann__
#define __RGBD_RF__eigenFlann__

#include <stdio.h>
#include <flann/flann.hpp>
#include <Eigen/Dense>
#include <vector>

using std::vector;

template <class T>
class EigenFlann
{
    flann::Index<flann::L2<T> > index_;    // store kd tree
    int dim_;
public:
    EigenFlann(const flann::IndexParams& params = flann::KDTreeIndexParams(4)):index_(params)
    {
        dim_ = 0;
    }
    ~EigenFlann()
    {
        
    }
    
    void setData(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data, const int trees = 4)
    {
        assert(data.rows() > 0);
        dim_ = (int)data.cols();
        
        flann::Matrix<T> dataset(const_cast<T *>(data.data()), (int)data.rows(), dim_);
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
        
        flann::Matrix<T> query_data_wrap(const_cast<T *>(query_data.data()), (int)query_data.rows(), dim_);
        index_.knnSearch(query_data_wrap, indices, dists, knn, flann::SearchParams(num_search_leaf));
    }
    
};

typedef EigenFlann<float> EigenFlann32F;
typedef EigenFlann<double> EigenFlann64F;

#endif /* defined(__RGBD_RF__eigenFlann__) */
