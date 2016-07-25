//
//  cvxUtil.hpp
//  RGB_RF
//
//  Created by jimmy on 2016-05-27.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef cvxUtil_cpp
#define cvxUtil_cpp

#include <stdio.h>
#include <vector>
#include <string>

using std::vector;
using std::string;

class CvxUtil
{
public:
    // generate random number in the range of [min_val, max_val]
    static vector<double>
    generateRandomNumbers(double min_val, double max_val, int rnd_num);    
    
    static inline bool isInside(const int width, const int height, const int x, const int y)
    {
        return x >= 0 && y >= 0 && x < width && y < height;
    }
    
    static void splitFilename (const string& str, string &path, string &file);
    
    template <typename T>
    static vector<size_t> sortIndices(const vector<T> &v) {
        
        // initialize original index locations
        vector<size_t> idx(v.size());
        for (size_t i = 0; i != idx.size(); ++i){
            idx[i] = i;
        }
        
        // sort indexes based on comparing values in v
        sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
        
        return idx;
    }
    
    // video input/ output
    static double millisecondsFromIndex(const int index, const double fps)
    {
        return index * 1000.0/fps;
    }
    

    
};


#endif /* cvxUtil_cpp */
