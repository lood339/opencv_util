//
//  eigen_geometry_util.cpp
//  PointLineReloc
//
//  Created by jimmy on 2017-05-05.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "eigen_geometry_util.h"

Eigen::Matrix3d EigenGeometryUtil::vector2SkewSymmetricMatrix(const Eigen::Vector3d & v)
{
    // https://en.wikipedia.org/wiki/Skew-symmetric_matrix
    Eigen::Matrix3d m = Eigen::Matrix3d::Zero();
    double a = v.x();
    double b = v.y();
    double c = v.z();
    m << 0, -c, b,
    c, 0, -a,
    -b, a, 0;
    
    return m;
}
