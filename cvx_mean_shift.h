//
//  cvx_mean_shift.h
//  PointLineReloc
//
//  Created by jimmy on 2017-04-14.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __PointLineReloc__cvx_mean_shift__
#define __PointLineReloc__cvx_mean_shift__

#include <stdio.h>
#include <Eigen/Dense>
#include <vector>

using std::vector;

class CvxMeanShift
{
public:
    struct Mode
    {
        Mode()
        {
            support = 0;
        }
        
        /**
         * @brief Compare modes by their support size.
         *
         * @return bool
         */
        bool operator<(const Mode& mode) const
        {
            return (this->support < mode.support);
        }
        
        void setSupportIndex(const vector<unsigned int>& indices) {
            support_indices = Eigen::VectorXi(indices.size());
            for( unsigned i = 0; i<indices.size(); i++) {
                support_indices[i] = (int)indices[i];
            }
        }
        
        Eigen::VectorXf mean;   // mean of the mode
        Eigen::MatrixXf covar;  // covariance matrix of the mode
        unsigned int support;       // number of samples that belong to this mode
        Eigen::VectorXi support_indices;
    };
    
    struct MeanShiftParameter
    {
    public:
        double band_width;  // large --> probability smooth
        int    min_size;    // mininum size required for a mode, small --> more mode
        double min_dis;     // The minimum distance required between modes, in each dimension. small --> more mode
        
        MeanShiftParameter()
        {
            band_width = 0.1;   // dis ^ 2 / band_width < min_dis  --> valide mode
            min_size   = 20;
            min_dis    = 1.0;    // relative distance to bandwith ?
        }
    };
    
public:
    static
    bool meanShift(const vector<Eigen::VectorXf>& data,
                   const MeanShiftParameter& param,
                   vector<CvxMeanShift::Mode> & modes);
    
    
    // sample part of data to estimate data density using mean shift
    static
    bool meanShift(const vector<Eigen::VectorXf>& data,
                   const MeanShiftParameter& param,
                   const vector<unsigned int>& data_index,
                   const int max_num_of_data,
                   vector<CvxMeanShift::Mode> & modes);
    
    
private:
    
    static
    Mode createMode(const Eigen::VectorXf& m,
                    const vector<Eigen::VectorXf>& input_data,
                    const vector<unsigned int>& support_index);
    
};

#endif /* defined(__PointLineReloc__cvx_mean_shift__) */
