//
//  Kabsch.h
//  RGB_RF
//
//  Created by jimmy on 2016-06-14.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef Kabsch_h
#define Kabsch_h

#include <Eigen/Geometry>

Eigen::Affine3d Find3DAffineTransform(Eigen::Matrix3Xd in, Eigen::Matrix3Xd out);

#endif /* Kabsch_h */
