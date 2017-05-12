//
//  vgl_algo.h
//  PointLineReloc
//
//  Created by jimmy on 2017-04-02.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __PointLineReloc__vgl_algo__
#define __PointLineReloc__vgl_algo__

#include <stdio.h>
#include "vgl_head_files.h"

class VglAlgo
{
public:
    // pixel locations along the linesegment,
    static void pixelAlongLineSegment(const vgl_line_segment_2d<double> & line_segment, const double area_width,
                                      const int w, const int h, 
                                      std::vector<vgl_point_2d<double> > & left_side_pts,
                                      std::vector<vgl_point_2d<double> > & right_side_pts);
    
    // pixel position along a thick line_segment 
    static bool lineSegmentPosition(const vgl_line_segment_2d<double> & line_segment, const double thickness,
                                    const int w, const int h,
                                    std::vector<vgl_point_2d<double> > & line_pts);
    
    // paralle shift 2D line segment
    static void parallelShiftLineSegment(const vgl_line_segment_2d<double> & initSeg, double distance,
                                         vgl_line_segment_2d<double> & seg1, vgl_line_segment_2d<double> & seg2);
    
private:
    static bool lineSegmentPositionUnsafe(const vgl_line_segment_2d<double> & segment, const double thickness,
                                          const int w, const int h, vcl_vector<vgl_point_2d<double> > & line_pts);

    
};


#endif /* defined(__PointLineReloc__vgl_algo__) */
