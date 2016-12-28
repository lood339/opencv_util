/** @file sift.h
 ** @brief SIFT (@ref sift)
 ** @author Andrea Vedaldi and Jimmy Chen
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

// modified from sift.h, 128 dimension --> 64 dimension, 4 x 4 x 4 (orientation)
#ifndef VL_SIFT_64_H
#define VL_SIFT_64_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <vl/sift.h>


/** @name Create
 ** @{
 **/
VL_EXPORT
VlSiftFilt*  vl_sift_new_64  (int width, int height,
                             int noctaves, int nlevels,
                             int o_min) ;


VL_EXPORT
void  vl_sift_calc_keypoint_descriptor_64   (VlSiftFilt *f,
                                          vl_sift_pix *descr,
                                          VlSiftKeypoint const* k,
                                          double angle) ;

VL_EXPORT
void  vl_sift_calc_raw_descriptor_64        (VlSiftFilt const *f,
                                          vl_sift_pix const* image,
                                          vl_sift_pix *descr,
                                          int widht, int height,
                                          double x, double y,
                                          double s, double angle0) ;

/*
 calculate the single orientation
 return at most one orientation
 */
VL_EXPORT
int   vl_sift_calc_keypoint_orientation  (VlSiftFilt *f,
                                          double angles [4],
                                          VlSiftKeypoint const*k);

#ifdef __cplusplus
}
#endif

/* VL_SIFT_64_H */
#endif
