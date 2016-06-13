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
    

    
};


#endif /* cvxUtil_cpp */
