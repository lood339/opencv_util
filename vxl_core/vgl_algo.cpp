//
//  vgl_algo.cpp
//  PointLineReloc
//
//  Created by jimmy on 2017-04-02.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "vgl_algo.h"
#include "vcl_head_files.h"



void VglAlgo::pixelAlongLineSegment(const vgl_line_segment_2d<double> & line_segment, const double area_width,
                                    const int w, const int h, 
                                    std::vector<vgl_point_2d<double> > & left_side_pts,
                                    std::vector<vgl_point_2d<double> > & right_side_pts)
{
    // parallel move the line segment
    double half_width = area_width/2.0;
    vgl_line_segment_2d<double> seg1;
    vgl_line_segment_2d<double> seg2;
    VglAlgo::parallelShiftLineSegment(line_segment, half_width, seg1, seg2);
    
    // pick up pixel position in the color profile area ( rectangular-like area)
    vcl_vector<vgl_point_2d<double> > pixels1;
    vcl_vector<vgl_point_2d<double> > pixels2;
    VglAlgo::lineSegmentPosition(seg2, half_width, w, h, left_side_pts);
    VglAlgo::lineSegmentPosition(seg1, half_width, w, h, right_side_pts);
}

bool VglAlgo::lineSegmentPosition(const vgl_line_segment_2d<double> & line_segment, const double thickness,
                                  const int w, const int h,
                                  std::vector<vgl_point_2d<double> > & linePts)
{
    vcl_vector<vgl_point_2d<double> > pts;
    VglAlgo::lineSegmentPositionUnsafe(line_segment, thickness, w, h, pts);
    for (int i = 0; i<pts.size(); i++) {
        vgl_point_2d<double> p = pts[i];
        if (p.x() >= 0 && p.x() < w && p.y() >= 0 && p.y() < h) {
            linePts.push_back(p);
        }
    }
    return true;
}

void VglAlgo::parallelShiftLineSegment(const vgl_line_segment_2d<double> & initSeg, double distance,
                                       vgl_line_segment_2d<double> & seg1, vgl_line_segment_2d<double> & seg2)
{
    // CCW rotated
    vgl_vector_2d<double> orthDir = rotated(initSeg.direction(), M_PI/2.0);
    orthDir = normalize(orthDir);
    
    vgl_point_2d<double> p1 = initSeg.point1();
    vgl_point_2d<double> p2 = initSeg.point2();
    
    vgl_vector_2d<double> dp = distance * orthDir;
    vgl_point_2d<double> p3(p1.x() + dp.x(), p1.y() + dp.y());
    vgl_point_2d<double> p4(p2.x() + dp.x(), p2.y() + dp.y());
    
    seg1 = vgl_line_segment_2d<double>(p3, p4);
    
    dp = -1.0 * dp;
    vgl_point_2d<double> p5(p1.x() + dp.x(), p1.y() + dp.y());
    vgl_point_2d<double> p6(p2.x() + dp.x(), p2.y() + dp.y());
    seg2 = vgl_line_segment_2d<double>(p5, p6);
}


static long int vil_vcl_round( double x )
{
    return (long int)( x > 0.0 ? x + 0.5 : x - 0.5 );
}

bool VglAlgo::lineSegmentPositionUnsafe(const vgl_line_segment_2d<double> & segment, const double thickness,
                                        const int w, const int h, vcl_vector<vgl_point_2d<double> > & linePts)
{
    vgl_point_2d< double > p1 = segment.point1();
    vgl_point_2d< double > p2 = segment.point2();
    
    double slope = vcl_abs( segment.slope_degrees() );
    
    if ( slope > 45.0 && slope < 135.0 )
    {
        int min_j = vcl_max(                 0, (int)vil_vcl_round( vcl_min( p1.y(), p2.y() ) ) );
        int max_j = vcl_min( (int)h-1,          (int)vil_vcl_round( vcl_max( p1.y(), p2.y() ) ) );
        
        for ( int j = min_j; j <= max_j; j++ )
        {
            if ( ( j < 0 ) || ( j >= (int)h ) ) continue;
            
            int i = (int)vil_vcl_round( -( segment.b() / segment.a() ) * j - ( segment.c() / segment.a() ) );
            
            if ( ( i < 0 ) || ( i >= (int)w ) )
            continue;
            
            //	hilitePixel( image, j, i, colour );
            linePts.push_back(vgl_point_2d<double>(i, j));
            for (unsigned int t = 1; t < thickness; t++)
            {
                if (i > 0 && i - t > 0 && i-t < (int)w){
                    //	hilitePixel( image, j, i-t, colour );
                    linePts.push_back(vgl_point_2d<double>(i-t, j));
                }
                if (i < (int)w && i+t < (int)w){
                    //	hilitePixel( image, j, i+t, colour );
                    linePts.push_back(vgl_point_2d<double>(i+t, j));
                }
            }
        }
        
    }
    else
    {
        int min_i = vcl_max(                 0, (int)vil_vcl_round( vcl_min( p1.x(), p2.x() ) ) );
        int max_i = vcl_min( (int)w-1,          (int)vil_vcl_round( vcl_max( p1.x(), p2.x() ) ) );
        
        for ( int i = min_i; i <= max_i; i++ )
        {
            if ( ( i < 0 ) || ( i >= (int)w ) ) continue;
            
            int j = (int)vil_vcl_round( -( segment.a() / segment.b() ) * i - ( segment.c() / segment.b() ) );
            
            if ( ( j < 0 ) || ( j >= (int)h ) ) continue;
            
            //	hilitePixel( image, j, i, colour );
            linePts.push_back(vgl_point_2d<double>(i, j));
            for (unsigned int t = 1; t < thickness; t++)
            {
                if (j > 0 && j-t > 0)
                {
                    
                    //	hilitePixel( image, j-t, i, colour );
                    linePts.push_back(vgl_point_2d<double>(i, j-t));
                }
                if (j < (int)h && j+t < (int)h)
                {
                    //	hilitePixel( image, j+t, i, colour );
                    linePts.push_back(vgl_point_2d<double>(i, j+t));
                }
            }
        }
    }
    
    return true;
}

