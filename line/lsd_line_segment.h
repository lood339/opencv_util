//
//  lsd_line_segment.h
//  PointLineReloc
//
//  Created by jimmy on 2017-03-29.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __PointLineReloc__lsd_line_segment__
#define __PointLineReloc__lsd_line_segment__

// LSD: a line segment detector
#include <stdio.h>
#include <vector>


using std::vector;

class LSD
{
public:
    struct LSDLineSegment2D
    {
        //    The seven values are:
        //    - x1,y1,x2,y2,width,p,-log10(NFA)
        //    for a line segment from coordinates (x1,y1) to (x2,y2),
        //    a width 'width', an angle precision of p in (0,1) given
        //    by angle_tolerance/180 degree, and NFA value 'NFA'.
        
        double x1, y1;
        double x2, y2;
        double width_;
        double angle_precision_;
        double NFA_; // ?
    };

    static
    void detectLines(const double * const image_data, int width, int height, std::vector<LSDLineSegment2D> & line_segments);
};


#endif /* defined(__PointLineReloc__lsd_line_segment__) */
