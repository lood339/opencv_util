//
//  lsd_line_segment.cpp
//  PointLineReloc
//
//  Created by jimmy on 2017-03-29.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "lsd_line_segment.h"
#include <stdlib.h>
#include <assert.h>
#include "lsd.h"

void LSD::detectLines(const double * const input_image_data, int width, int height, std::vector<LSDLineSegment2D> & line_segments)
{
    double * imageData = NULL;
    double * out = NULL;
    int n = 0;
    imageData = (double *) malloc( width * height * sizeof(double) );
    assert(imageData);
    memcpy(imageData, input_image_data, width * height * sizeof(double));
    
    /* LSD call */
    out = lsd(&n, imageData, width, height);
    
    /* print output */
    // printf("%d line segments found:\n",n);
    line_segments.resize(n);
    for(int i=0; i<n; i++)
    {
        double x1 = out[7*i + 0];
        double y1 = out[7*i + 1];
        double x2 = out[7*i + 2];
        double y2 = out[7*i + 3];
        double line_width = out[7*i + 4];
        double p   = out[7*i + 5];
        double tmp = out[7*i + 6];
        
        line_segments[i].x1 = x1;
        line_segments[i].y1 = y1;
        line_segments[i].x2 = x2;
        line_segments[i].y2 = y2;
        line_segments[i].width_ = line_width;
        line_segments[i].angle_precision_ = p;
        line_segments[i].NFA_ = tmp;
    }
    free( (void *) imageData );
    free( (void *) out );
}