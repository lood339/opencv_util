//
//  cvxMeanShift.h
//  RGB_RF
//
//  Created by jimmy on 2016-07-15.
//  Copyright (c) 2016 jimmy. All rights reserved.
//

#ifndef __RGB_RF__cvxMeanShift__
#define __RGB_RF__cvxMeanShift__

// wrap meanshift from "Nonlinear Mean Shift over Riemannian Manifolds " IJCV 2009

#include <stdio.h>
#include <stdio.h>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <unordered_map>
#include <string>

using std::vector;
using std::unordered_map;
using std::string;

class CvxMeanShiftParameter;

class CvxMeanShift
{
public:
    // data: every row is a point
    // modes: output row vector
    // wt: weight of each mode
    static int meanShift(const cv::Mat & data,
                         vector<cv::Mat> & modes,
                         vector<double> & wt,
                         const CvxMeanShiftParameter & param);
    
    // only keep largest mode
    static bool
    meanShiftLargestMode(const cv::Mat & data,
                         cv::Mat & mode,
                         double & wt,
                         const CvxMeanShiftParameter & param);
    
};

class CvxMeanShiftParameter
{
public:
    double band_width_;  // large --> probability smooth
    int min_size_;    // mininum size required for a mode, small --> more mode
    double min_dis_;  // The minimum distance required between modes, small --> more mode
    int max_mode_num_;
    
    CvxMeanShiftParameter()
    {
        band_width_ = 0.1;   // dis ^ 2 / band_width < min_dis  --> valide mode
        min_size_   = 20;
        min_dis_    = 0.1;  // relative distance to bandwith ?
        max_mode_num_ = 200;  //maximum mode numbers in the cluster
    }
    
    CvxMeanShiftParameter(const double band_width, const int min_size, const double min_dis)
    {
        band_width_ = band_width;
        min_size_ = min_size;
        min_dis_ = min_dis;
        max_mode_num_ = 200;
    }
    
    bool readFromFile(const char* file_name)
    {
        FILE *pf = fopen(file_name, "r");
        if (!pf) {
            printf("Error: can not open %s \n", file_name);
            return false;
        }
        
        const double param_num = 3;
        unordered_map<std::string, double> imap;
        for(int i = 0; i<param_num; i++)
        {
            char s[1024] = {NULL};
            double val = 0.0;
            int ret = fscanf(pf, "%s %lf", s, &val);
            if (ret != 2) {
                break;
            }
            imap[string(s)] = val;
        }
        assert(imap.size() == 7);
        fclose(pf);
        
        band_width_ = imap[string("band_width")];
        min_size_ = (int)imap[string("min_size")];
        min_dis_ = imap[string("min_dis")];
        return true;
    }
    
    void printSelf()
    {
        printf("mean shift parameters: \n");
        printf("band   width: %f\n", band_width_);
        printf("min     size: %d\n", min_size_);
        printf("min distance: %f\n\n", min_dis_);
    }
    
};

#endif /* defined(__RGB_RF__cvxMeanShift__) */
