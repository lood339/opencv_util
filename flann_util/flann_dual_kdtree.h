//
//  flann_dual_kdtree.h
//  OLNN
//
//  Created by jimmy on 2017-01-02.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef OLNN_flann_dual_kdtree_h
#define OLNN_flann_dual_kdtree_h

#include <flann/flann.hpp>
#include <Eigen/Dense>
#include <vector>

using std::vector;

template <class T>
class FlannDualKDTree
{
    flann::Index<flann::L2<T> > index_;      // store kd tree
    int dim_;
public:
    FlannDualKDTree(const flann::IndexParams& params = flann::DualKDtreeIndexParams(-1, 128, 1)):index_(params)
    {
        dim_ = 0;
    }
    
    ~FlannDualKDTree()
    {
        
    }
    
    // auxilary_div_depth: -1, no auxilary dimension
    void setData(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data, const int trees,
                 const int auxilary_div_depth = -1,
                 const int feat_veclen = 128)
    {
        assert(data.rows() > 0);
        dim_ = (int)data.cols();
        assert(dim_ >= feat_veclen);
        
        flann::Matrix<T> dataset(const_cast<T *>(data.data()), (int)data.rows(), dim_);
        index_ = flann::Index< flann::L2<T> >(dataset, flann::DualKDtreeIndexParams(auxilary_div_depth, feat_veclen, trees));
        index_.buildIndex();
    }
    
    // auxilary_eps: hard constraint
    void search(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & query_data,
                vector<vector<int> > & indices,
                vector<vector<T> > & dists,
                const int knn,
                int num_search_leaf = 128,
                const float auxilary_eps = 100) const
    {
        assert(query_data.rows() > 0);
        assert(dim_ == query_data.cols());
        
        flann::Matrix<T> query_data_wrap(const_cast<T *>(query_data.data()), (int)query_data.rows(), dim_);
        
        flann::SearchParams search_param(num_search_leaf);
        search_param.auxilary_eps = auxilary_eps;
        index_.knnSearch(query_data_wrap, indices, dists, knn, search_param);
    }
    
};

typedef FlannDualKDTree<float>  FlannDualKDTree32F;
typedef FlannDualKDTree<double> FlannDualKDTree64F;


#endif
