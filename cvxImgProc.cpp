//
//  cvxImgProc.cpp
//  RGB_RF
//
//  Created by jimmy on 2016-06-02.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "cvxImgProc.hpp"
#include "cv_draw.hpp"

using cv::Rect;

Mat CvxImgProc::gradientOrientation(const Mat & img, const int gradMagThreshold)
{
    assert(img.type() == CV_8UC1 || img.type() == CV_8UC3);
    
    Mat src_gray;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    
    GaussianBlur( img, img, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
    /// Convert it to gray
    if (img.channels() == 3) {
        cvtColor( img, src_gray, CV_BGR2GRAY );
    }
    else {
        src_gray = img;
    }
    
    /// Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Mat grad;
    /// Gradient X
    Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, cv::BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );
    /// Gradient Y
    Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, cv::BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );
    
    grad_x.convertTo(grad_x, CV_64FC1);
    grad_y.convertTo(grad_y, CV_64FC1);
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    
    cv::threshold(grad, grad, gradMagThreshold, 1.0, cv::THRESH_BINARY); // grad 0 or 1
    grad.convertTo(grad, CV_64FC1);
    
    Mat orientation;
    cv::phase(grad_x, grad_y, orientation, false);
    
    Mat threshold_orientation = orientation.mul(grad); // set small gradient positions as zero, other graduent as the same
    
 //   cv::Mat vmat = CvDraw::visualize_gradient(grad, threshold_orientation);
 //   cv::imshow("orientation", vmat);
 //   cv::waitKey();
    
    return threshold_orientation;
}

static void CentroidOrientationICAngles(const Mat& img,
                                        const std::vector<cv::Point2d>& pts,
                                        const std::vector<int> & u_max,
                                        int half_k,
                                        std::vector<float> & angles)
{
    assert(img.channels() == 1);
    assert(img.type() == CV_8UC1);
    
    int step = (int)img.step1();
    angles.resize(pts.size());
    
    for(size_t ptidx = 0; ptidx < pts.size(); ptidx++ )
    {
    //    const Rect& layer = layerinfo[pts[ptidx].octave];
    //    const uchar* center = &img.at<uchar>(cvRound(pts[ptidx].pt.y) + layer.y, cvRound(pts[ptidx].pt.x) + layer.x);
        int y = cvRound(pts[ptidx].y);
        int x = cvRound(pts[ptidx].x);
        if (x <= half_k || x + half_k >= img.cols ||
            y <= half_k || y + half_k >= img.rows)
        {
            angles[ptidx] = 0.0f; // out of boundary samples
            continue;
        }
                        
        const uchar* center = &img.at<uchar>(y, x);
        int m_01 = 0, m_10 = 0;
        
        // Treat the center line differently, v=0
        for (int u = -half_k; u <= half_k; ++u){
            m_10 += u * center[u];
        }
        
        // Go line by line in the circular patch
        for (int v = 1; v <= half_k; ++v)
        {
            // Proceed over the two lines
            int v_sum = 0;
            int d = u_max[v];
            for (int u = -d; u <= d; ++u)
            {
                int val_plus = center[u + v*step], val_minus = center[u - v*step];
                v_sum += (val_plus - val_minus);
                m_10 += u * (val_plus + val_minus);
            }
            m_01 += v * v_sum;
        }
        angles[ptidx] = cv::fastAtan2((float)m_01, (float)m_10);
    }
}

void CvxImgProc::centroidOrientation(const Mat & img, const vector<cv::Point2d> & pts,
                                     const int patchSize, vector<float> & angles)
{
    assert(img.channels() == 1);
    assert(img.type() == CV_8UC1);
    
    // pre-compute the end of a row in a circular patch
    int halfPatchSize = patchSize / 2;
    std::vector<int> umax(halfPatchSize + 2);
    
    int v, v0, vmax = cvFloor(halfPatchSize * std::sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(halfPatchSize * std::sqrt(2.f) / 2);
    for (v = 0; v <= vmax; ++v){
        umax[v] = cvRound(std::sqrt((double)halfPatchSize * halfPatchSize - v * v));
    }
    
    // Make sure we are symmetric
    for (v = halfPatchSize, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }
    
    // debug
    //printf("centroid orientation umax: ");
    //for (int i = 0; i<umax.size(); i++) {
    //    printf("%d ", umax[i]);
    //}
    //printf("\n");
    
    CentroidOrientationICAngles(img, pts, umax, halfPatchSize, angles);
}

void CvxImgProc::centroidOrientation(const Mat & img, const int patchSize, const int smoothSize, Mat & orientation)
{
    vector<cv::Point2d> pts;
    for (int r = 0; r<img.rows; r++) {
        for (int c = 0; c<img.cols; c++) {
            pts.push_back(cv::Point2d(c, r));
        }
    }
    
    vector<float> angles;
    CvxImgProc::centroidOrientation(img, pts, patchSize, angles);
    
    orientation = cv::Mat::zeros(img.rows, img.cols, CV_64FC1);
    for (int r = 0; r<orientation.rows; r++) {
        for (int c = 0; c<orientation.cols; c++) {
            orientation.at<double>(r, c) = angles[r * orientation.cols + c];
        }
    }
    
    cv::GaussianBlur(orientation, orientation, cv::Size(smoothSize, smoothSize), 0.0, 0.0);
}






