//
//  flann_kmeans.h
//  OLNN
//
//  Created by jimmy on 2016-12-22.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef OLNN_flann_kmeans_h
#define OLNN_flann_kmeans_h

#include <flann/flann.hpp>
#include <Eigen/Dense>
#include <vector>

using std::vector;

template <class T>
class FlannKmeans
{
    flann::Index<flann::L2<T> > index_;    // store kmeans index
    int dim_;
public:
    FlannKmeans(const flann::IndexParams& params = flann::KMeansIndexParams(32, 10)):index_(params)
    {
        dim_ = 0;
    }
    ~FlannKmeans()
    {
        
    }
    
    void setData(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data, const int branching, const int iterations = 10)
    {
        assert(data.rows() > 0);
        dim_ = (int)data.cols();
        
        flann::Matrix<T> dataset(const_cast<T *>(data.data()), (int)data.rows(), dim_);
        index_ = flann::Index< flann::L2<T> >(dataset, flann::KMeansIndexParams(branching, iterations));
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
    
    void loadIndex(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data,
                   const char *file_name)
    {
        
        flann::SavedIndexParams save_param(file_name);
        dim_ = (int)data.cols();
        flann::Matrix<T> dataset(const_cast<T *>(data.data()), (int)data.rows(), dim_);
        index_ = flann::Index< flann::L2<T> >(dataset, save_param);
    }
    
    void saveIndex(const char *file_name)
    {
        index_.save(std::string(file_name));
        printf("save to %s\n", file_name);
    }

    
};

typedef FlannKmeans<float> FlannKmeans32F;
typedef FlannKmeans<double> FlannKmeans64F;

#endif
