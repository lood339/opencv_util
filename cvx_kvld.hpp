//
//  cvx_kvld.hpp
//  MAVGoogleImageMatching
//
//  Created by jimmy on 2015-11-28.
//  Copyright © 2015 jimmy. All rights reserved.
//

#ifndef cvx_kvld_cpp
#define cvx_kvld_cpp

#include "vxl_head_file.h"
#include <bapl/bapl_keypoint_set.h>


struct cvx_kvld_parameter
{
    vcl_vector<vcl_pair<int, int> > matches;
    vcl_vector<bapl_keypoint_sptr> keypoint_1;
    vcl_vector<bapl_keypoint_sptr> keypoint_2;
};

class cvx_kvld
{
public:
    static bool kvld_matching(const vil_image_view<vxl_byte> & image1,
                              const vil_image_view<vxl_byte> & image2,
                              vcl_vector<bapl_key_match> & final_matches,
                              vcl_vector<bool> & is_valid,
                              const cvx_kvld_parameter & param);
    
                       
};

#endif /* cvx_kvld_cpp */
