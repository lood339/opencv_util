//
//  eigen_least_square.h
//  Classifer_RF
//
//  Created by jimmy on 2017-03-13.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__eigen_least_square__
#define __Classifer_RF__eigen_least_square__

#include <stdio.h>
#include <vector>
#include <map>

using std::vector;
using std::map;

class EigenLeastSquare
{
public:
    // solve Ax = b
    static bool solver(vector<map<int, double> > & A, vector<double> & b, bool overConstraint, int var_Num, double *result);

};

#endif /* defined(__Classifer_RF__eigen_least_square__) */
