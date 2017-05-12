//
//  eigen_util.cpp
//  Classifer_RF
//
//  Created by jimmy on 2017-03-13.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "eigen_util.h"
#include <map>
#include <assert.h>
#include "eigen_least_square.h"

using std::map;

bool EigenUtil::L2NormSmooth(const vector<double> & data, vector<double> & smoothedData, double w1, double w2, double w3)
{
    vector<map<int, double> > leftVec;
    vector<double> rightVec;
    for (int i = 0; i<data.size(); i++) {
        // constraint to original signal
        {
            map<int, double> imap;
            double right_val = data[i] * w1;
            imap[i] = 1.0 * w1;
            
            leftVec.push_back(imap);
            rightVec.push_back(right_val);
        }
        
        // constraint on velocity
        {
            if (i > 0) {
                map<int, double> imap;
                double right_val = 0.0;
                imap[i]   =  1.0 * w2;
                imap[i-1] = -1.0 * w2;
                
                leftVec.push_back(imap);
                rightVec.push_back(right_val);
            }
        }
        
        // constraint on acceleration
        {
            if (i > 0 && i<data.size() - 1) {
                map<int, double> imap;
                double right_val = 0.0;
                imap[i+1] =  1.0 * w3;
                imap[i]   = -2.0 * w3;
                imap[i-1] =  1.0 * w3;
                
                leftVec.push_back(imap);
                rightVec.push_back(right_val);
            }
        }
    }
    assert(leftVec.size() == rightVec.size());
    
    smoothedData.resize(data.size());
    bool isSolved = EigenLeastSquare::solver(leftVec, rightVec, true, (int)smoothedData.size(), &smoothedData[0]);
    assert(isSolved);
    return true;
}

bool EigenUtil::L2NormSommth(const vector<double> & data, const vector<int> time_stamp,
                             vector<double> & smoothed_data, const double w1, const double w2, const double w3)
{
    assert(data.size() == time_stamp.size());
    for (int i = 1; i<time_stamp.size(); i++) {
        assert(time_stamp[i-1] <= time_stamp[i]);
    }
    vector<map<int, double> > leftVec;
    vector<double> rightVec;
    for (int i = 0; i<data.size(); i++) {
        // constraint to original signal
        {
            map<int, double> imap;
            double right_val = data[i] * w1;
            imap[i] = 1.0 * w1;
            
            leftVec.push_back(imap);
            rightVec.push_back(right_val);
        }
        
        // constraint on velocity
        {
            if (i > 0) {
                map<int, double> imap;
                int step = time_stamp[i] - time_stamp[i-1];
                if (step == 0) {
                    step = 1;  // approximate
                }
                double cur_wt = w2/step;
                double right_val = 0.0;
                imap[i]   =  1.0 * cur_wt;
                imap[i-1] = -1.0 * cur_wt;
                
                leftVec.push_back(imap);
                rightVec.push_back(right_val);
            }
        }
        
        // constraint on acceleration
        {
            if (i > 0 && i<data.size() - 1) {
                map<int, double> imap;
                int step = time_stamp[i+1] - time_stamp[i-1];
                if (step == 0) {
                    step = 1;
                }
                double cur_wt = w3/step;
                double right_val = 0.0;
                imap[i+1] =  1.0 * cur_wt;
                imap[i]   = -2.0 * cur_wt;
                imap[i-1] =  1.0 * cur_wt;
                
                leftVec.push_back(imap);
                rightVec.push_back(right_val);
            }
        }
    }
    assert(leftVec.size() == rightVec.size());
    
    smoothed_data.resize(data.size());
    bool isSolved = EigenLeastSquare::solver(leftVec, rightVec, true, (int)smoothed_data.size(), &smoothed_data[0]);
    assert(isSolved);
    return true;
}
