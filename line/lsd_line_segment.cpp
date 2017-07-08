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
#include <math.h>
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

static vector<LSD::LSDLineSegment2D> getHalfSegment(const LSD::LSDLineSegment2D& line, const double max_length)
{
    double x1 = line.x1;
    double y1 = line.y1;
    double x2 = line.x2;
    double y2 = line.y2;
    //double dist = cv::norm(p1 - p2);
    double dx = x2 - x1;
    double dy = y2 - y1;
    double dist = sqrt(dx * dx + dy * dy);
    if (dist <= max_length) {
        vector<LSD::LSDLineSegment2D> lines;
        lines.push_back(line);
        return lines;
    }
    else {
        //cv::Point2d mid_p = (p1 + p2)/2.0;
        double mid_x = (x1 + x2)/2.0;
        double mid_y = (y1 + y2)/2.0;
        LSD::LSDLineSegment2D line1 = line;
        line1.x2 = mid_x;
        line1.y2 = mid_y;
        LSD::LSDLineSegment2D line2 = line;
        line2.x1 = mid_x;
        line2.y1 = mid_y;
        
        vector<LSD::LSDLineSegment2D> left_lines = getHalfSegment(line1, max_length);
        vector<LSD::LSDLineSegment2D> right_lines = getHalfSegment(line2, max_length);
        assert(left_lines.size() >= 1 && right_lines.size() >= 1);
        
        left_lines.insert(left_lines.end(), right_lines.begin(), right_lines.end());
        return left_lines;
    }
}

void LSD::shortenLineSegments(std::vector<LSD::LSDLineSegment2D>& lines, double max_length)
{
    vector<LSD::LSDLineSegment2D> short_lines;
    for (int i = 0; i<lines.size(); i++) {
        vector<LSD::LSDLineSegment2D> s_lines = getHalfSegment(lines[i], max_length);
        short_lines.insert(short_lines.end(), s_lines.begin(), s_lines.end());
    }
    lines = short_lines;
}
