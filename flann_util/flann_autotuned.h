//
//  flann_autotuned.h
//  OLNN
//
//  Created by jimmy on 2016-12-07.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef OLNN_flann_autotuned_h
#define OLNN_flann_autotuned_h


#include <stdio.h>
#include <flann/flann.hpp>
#include <Eigen/Dense>
#include <vector>
#include <time.h>

using std::vector;

template <class T>
class FlannAutotuned
{
    flann::Index<flann::L2<T> > index_;    // store kd tree
    int dim_;
public:
    FlannAutotuned(const flann::IndexParams& params = flann::AutotunedIndexParams(0.8)):index_(params)
    {
        dim_ = 0;
    }
    
    ~FlannAutotuned()
    {
        
    }
    
    void setData(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data, const double target_precision = 0.8)
    {
        assert(data.rows() > 0);
        dim_ = (int)data.cols();
        
        flann::Matrix<T> dataset(const_cast<T *>(data.data()), (int)data.rows(), dim_);
        index_ = flann::Index< flann::L2<T> >(dataset, flann::AutotunedIndexParams(target_precision, 1.0));
        index_.buildIndex();
    }
    
    void search(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & query_data,
                vector<vector<int> > & indices,
                vector<vector<T> > & dists,
                const int knn,
                int num_search_leaf = flann::FLANN_CHECKS_AUTOTUNED) const
    {
        assert(query_data.rows() > 0);
        assert(dim_ == query_data.cols());
        
        flann::Matrix<T> query_data_wrap(const_cast<T *>(query_data.data()), (int)query_data.rows(), dim_);
        index_.knnSearch(query_data_wrap, indices, dists, knn, flann::SearchParams(num_search_leaf));
    }
};

typedef FlannAutotuned<float>  FlannAutotuned32F;
typedef FlannAutotuned<double> FlannAutotuned64F;

#endif
