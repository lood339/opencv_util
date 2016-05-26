//
//  vxl_asift_warp.hpp
//  MAVGoogleImageMatching
//
//  Created by jimmy on 2015-10-30.
//  Copyright © 2015 jimmy. All rights reserved.
//

#ifndef vxl_asift_warp_cpp
#define vxl_asift_warp_cpp

// wrap ASIFT from open cv to vxl
#include <vnl/vnl_matrix_fixed.h>
#include <vil/vil_image_view.h>

class vxl_asift_warp
{
public:
    // rotate: [-180, 180] degree
    // tilt  : [0, 90], along y = h/2 axis degree 45, 60
    // warped_image: result
    // return: affine matrix applited to the image (not done yet)
    static vnl_matrix_fixed<double, 2, 3> warp_image(const vil_image_view<vxl_byte> & image,
                                                     double rotate,
                                                     double tilt,
                                                     vil_image_view<vxl_byte> & warped_image);
};

#endif /* vxl_asift_warp_cpp */
