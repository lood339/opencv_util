//
//  eigen_geometry_util.h
//  PointLineReloc
//
//  Created by jimmy on 2017-05-05.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __PointLineReloc__eigen_geometry_util__
#define __PointLineReloc__eigen_geometry_util__

#include <stdio.h>
#include <Eigen/Dense>

class EigenGeometryUtil
{
public:
    
    static Eigen::Matrix3d vector2SkewSymmetricMatrix(const Eigen::Vector3d & v);
    
    
    
};

#endif /* defined(__PointLineReloc__eigen_geometry_util__) */
