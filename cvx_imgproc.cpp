//
//  cvxImgProc.cpp
//  RGB_RF
//
//  Created by jimmy on 2016-06-02.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "cvx_imgproc.hpp"
#include "cvx_draw.hpp"
#include <numeric>

using cv::Rect;
using cv::Vec3b;
using cv::Vec6f;
using cv::Mat;

void CvxImgProc::imageGradient(const Mat & color_img, Mat & grad)
{
    assert(color_img.type() == CV_8UC3);
    
    const int scale = 1;
    const int delta = 0;
    const int ddepth = CV_16S;
    
    cv::Mat gray;
    cv::GaussianBlur( color_img, color_img, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
    
    /// Convert it to gray
    cv::cvtColor( color_img, gray, CV_BGR2GRAY );
    
    /// Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    
    /// Gradient X
    Sobel( gray, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );
    
    /// Gradient Y
    Sobel( gray, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );
    
    /// Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
    assert(grad.type() == CV_8UC1);
    
}

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

Mat CvxImgProc::groupPatches(const vector<cv::Mat> & patches, int colNum)
{
    assert(patches.size() > 0);
    
    int rowNum = (int)patches.size()/colNum;
    if (patches.size()%colNum != 0) {
        rowNum++;
    }
    
    int patch_w = patches[0].cols;
    int patch_h = patches[0].rows;
    
    Mat img = cv::Mat::zeros(rowNum * patch_h, colNum * patch_w, patches[0].type());
    for (int i = 0; i<patches.size(); i++) {
        int r = i / colNum;
        int c = i % colNum;
        int sy = r * patch_h;
        int sx = c * patch_w;
        patches[i].copyTo(img(cv::Rect(sx, sy, patch_w, patch_h)));
    }
    return img;
}


bool CvxImgProc::estimateLineOrientation(const Mat& gry_img, const cv::Point2d & startPoint,
                                         const cv::Point2d & endPoint,const int line_width)
{
    assert(gry_img.type() == CV_8UC1);
    
    cv::Point pt1(startPoint.x, startPoint.y);
    cv::Point pt2(endPoint.x, endPoint.y);
    
    cv::LineIterator it(gry_img, pt1, pt2, 8);
    
    vector<unsigned char> buf(it.count);
    
    for(int i = 0; i < it.count; i++, ++it){
     //   buf[i] = *it;
    }
    return true;
}

namespace cvx {
    struct greaterThanPtr :
    public std::binary_function<const float *, const float *, bool>
    {
        bool operator () (const float * a, const float * b) const
        // Ensure a fully deterministic result of the sort
        { return (*a > *b) ? true : (*a < *b) ? false : (a > b); }
    };

    void goodFeaturesToTrack( cv::InputArray _image, cv::OutputArray _corners,
                             int maxCorners, double qualityLevel, double minDistance,
                             cv::InputArray _mask, int blockSize,
                             bool useHarrisDetector, double harrisK )
    {
        Mat image = _image.getMat();
        
        /// My Harris matrix -- Using cornerEigenValsAndVecs
        Mat corner_dst = Mat::zeros( image.size(), CV_32FC(6) );
        Mat eig = Mat::zeros( image.size(), CV_32FC1 );  // corner strength
        Mat tmp;
        
        // this part is from OpenCV, goodFeaturesToTrack
        cornerHarris( image, eig, blockSize, 3, harrisK);
        double maxVal = 0;
        minMaxLoc( eig, 0, &maxVal, 0, 0, _mask );        
        cv::threshold(eig, eig, maxVal * qualityLevel, 0, cv::THRESH_TOZERO );
        cv::dilate( eig, tmp, Mat());
        
        cv::Size imgsize = image.size();
        std::vector<const float*> tmp_corners;
        
        // collect list of pointers to features - put them into temporary image
        Mat mask = _mask.getMat();
        for( int y = 1; y < imgsize.height - 1; y++ )
        {
            const float* eig_data = (const float*)eig.ptr(y);
            const float* tmp_data = (const float*)tmp.ptr(y);
            const uchar* mask_data = mask.data ? mask.ptr(y) : 0;
            
            for( int x = 1; x < imgsize.width - 1; x++ )
            {
                float val = eig_data[x];
                if( val != 0 && val == tmp_data[x] && (!mask_data || mask_data[x]) )
                    tmp_corners.push_back(eig_data + x);
            }
        }
        
        size_t i, j, total = tmp_corners.size();
        
        if (total == 0)
        {
            printf("warning: find zero cornders\n");
            _corners.release();
            return;
        }
        
        std::vector<cv::Point2f> corners(total);
        std::sort(tmp_corners.begin(), tmp_corners.end(), greaterThanPtr());
        for( i = 0; i < total; i++ )
        {
            int ofs = (int)((const uchar*)tmp_corners[i] - eig.ptr());
            int y = (int)(ofs / eig.step);
            int x = (int)((ofs - y*eig.step)/sizeof(float));
            
            corners[i] = cv::Point2f((float)x, (float)y);
        }
        
        if (total <= maxCorners) {
            Mat(corners).convertTo(_corners, _corners.fixedType() ? _corners.type() : CV_32F);
            return;
        }
        
        // adaptive non-maximal suppression
        float diag = sqrt(imgsize.width * imgsize.width + imgsize.height * imgsize.height);
        float c = 0.9;   // sufficiently larger parameter
        std::vector<float> max_radius(total, diag);
        for (i = 1; i < total; i++) {
            // for each corner, update maximum radius
            float s1 = *tmp_corners[i];  // corner strength
            for (j = 0; j<i; j++) {
                float r = cv::norm(corners[i] - corners[j]); // radius
                float s2 = *tmp_corners[j];
                if (r < max_radius[i] &&
                    s1 < c * s2) {
                    max_radius[i] = r;
                }
            }
        }
        
        // sort by maximum radius, not by the corner strength
        std::vector<size_t> index(max_radius.size());
        std::iota(index.begin(), index.end(), 0);
        // sort index by the value in max_radius
        std::sort(index.begin(), index.end(), [&max_radius](size_t i1, size_t i2){return max_radius[i1] > max_radius[i2];});
        assert(index.size() > maxCorners);
        
        //float threshold_raidus = max_radius[index[maxCorners]];
        //printf("thresholding radius is %lf\n", threshold_raidus);
        for (i = 0; i < maxCorners; i++) {
            size_t idx = index[i];
            if (idx != i) {
                std::swap(corners[i], corners[idx]);
            }
        }
        corners.resize(maxCorners);
        
        Mat(corners).convertTo(_corners, _corners.fixedType() ? _corners.type() : CV_32F);
        return;
    }
    
    void trackLineSegmentCenter(const vector<cv::Vec4f>& _src,
                                const vector<cv::Vec4f>& _dst,
                                vector<cv::Vec2f>& _centers,
                                const cv::Size& sz,
                                cv::OutputArray center_patch_distance,
                                cv::OutputArray _dist_map,
                                double search_length, int block_size)
    {
        assert(block_size%2 == 1);
        assert(_centers.size() == 0);
        assert(_src.size() > 0);
        assert(_dst.size() > 0);
        
        const int cols = sz.width;
        const int rows = sz.height;
        
        // 1. create a distance map in dst image
        Mat bw = Mat(rows, cols, CV_8UC1, cv::Scalar(255));
        Mat dist_map;
        for (int i = 0; i<_dst.size(); i++) {
            cv::line(bw, cv::Point(_dst[i][0], _dst[i][1]), cv::Point(_dst[i][2], _dst[i][3]),
                     cv::Scalar::all(0), 1, 0);
        }
       
        //imshow("Binary Image", bw);
        // Perform the distance transform algorithm
        // point on edges has zero distance
        cv::distanceTransform(bw, dist_map, CV_DIST_L2, 3);
        assert(dist_map.type() == CV_32FC1);
        if (_dist_map.needed()) {
            dist_map.copyTo(_dist_map);
        }        
        
        // 2. search correspondence of each src point center in the distance map
        Mat src_edge_map = Mat(rows, cols, CV_8UC1, cv::Scalar(0));
        for (int i = 0; i<_src.size(); i++) {
            cv::line(src_edge_map, cv::Point(_src[i][0], _src[i][1]), cv::Point(_src[i][2], _src[i][3]),
                     cv::Scalar::all(255), 1, 0);
        }
        
        // 3. the one with minimum distance
        const int half_block_size = block_size/2;
        vector<float> min_patch_distances;
        vector<int> out_of_image_index;
        for (int i = 0; i<_src.size(); i++) {
            // https://stackoverflow.com/questions/8664866/draw-perpendicular-line-to-a-line-in-opencv
            cv::Point2f p1(_src[i][0], _src[i][1]);
            cv::Point2f p2(_src[i][2], _src[i][3]);
            cv::Point2f mid_p = (p1 + p2)/2.0;
            cv::Point2f v(p2.x - p1.x, p2.y - p1.y);
            v /= sqrt(v.x * v.x + v.y * v.y);
            std::swap(v.x, v.y);
            v.x *= -1.0; // clockwise rotation
            
            cv::Point2f p3 = mid_p + search_length * v;
            cv::Point2f p4 = mid_p - search_length * v;
            cv::Point p3_img(cvRound(p3.x), cvRound(p3.y));
            cv::Point p4_img(cvRound(p4.x), cvRound(p4.y));
            cv::clipLine(sz, p3_img, p4_img);
            cv::Point mid_p_img(cvRound(mid_p.x), cvRound(mid_p.y));
            
            double min_dist = INT_MAX;
            if (mid_p_img.x - half_block_size < 0 ||
                mid_p_img.y - half_block_size < 0 ||
                mid_p_img.x + half_block_size >= cols ||
                mid_p_img.y + half_block_size >= rows) {
                _centers.push_back(mid_p);   // default answer, has no movement
                min_patch_distances.push_back(-1);  // set a small number
                out_of_image_index.push_back(i);
                continue;
            }
            
            cv::Mat patch = src_edge_map(cv::Rect(mid_p_img.x - half_block_size,
                                                  mid_p_img.y - half_block_size, block_size, block_size));
            int edge_pixel_num = cv::countNonZero(patch);
            assert(edge_pixel_num > 0);
            
            // sample along the normal direction of the line
            cv::LineIterator it(bw, p3_img, p4_img, 8);
            cv::Point min_pos = mid_p_img;   // default,
            //vector<double> cur_distances_debug;
            for(int ct = 0; ct < it.count; ct++, ++it)
            {
                cv::Point p = it.pos();  // center location of the patch
                // is the patch centered in p inside the image ?
                if (p.x - half_block_size < 0 ||
                    p.y - half_block_size < 0 ||
                    p.x + half_block_size >= cols ||
                    p.y + half_block_size >= rows){
                    continue;
                }
                
                float dist = 0;
                for (int j = 0; j < block_size; j++) {
                    for (int k = 0; k < block_size; k++) {
                        if (patch.at<unsigned char>(j, k) == 255) {
                            int x = p.x - half_block_size + k;
                            int y = p.y - half_block_size + j;
                            dist += dist_map.at<float>(y, x);
                        }
                    } // k
                } //j
                //printf("%lf \t", dist);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_pos = p;
                }
                //cur_distances_debug.push_back(dist);
            }
            _centers.push_back(cv::Point2f(min_pos.x, min_pos.y));
            min_patch_distances.push_back(min_dist/edge_pixel_num);
            
            // debug
            //printf("patch number: %d\n", i);
            //for (int j = 0; j<cur_distances_debug.size(); j++) {
            //    printf("%lf ", cur_distances_debug[j]);
            //}
            //printf("\n");
            if (i == 24) {
                //cv::imwrite("patch.png", patch);
            }
        }
        assert(_centers.size() == _src.size());
        assert(_centers.size() == min_patch_distances.size());
        // 4. return result
        if (center_patch_distance.needed()) {
            float max_v = *std::max(min_patch_distances.begin(), min_patch_distances.end());
            // fill the missing value as max value
            for (int i = 0; i<out_of_image_index.size(); i++) {
                min_patch_distances[out_of_image_index[i]] = max_v;
            }
            Mat(min_patch_distances).copyTo(center_patch_distance);
        }
    }   
}






