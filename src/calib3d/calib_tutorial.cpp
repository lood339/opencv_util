#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>

using namespace std;
using namespace cv;

int main() 
{   
    string path("/Users/jimmy/Code/opencv_util/data/");
    vector<string> left_names;
    vector<string> right_names;

    /* 
    auto zero_lead = [](const int value, const unsigned precision) {
        ostringstream oss;
        oss<<std::setw(precision)<<std::setfil('0')<<value;
        return oss.str();
    }
    */

   int num = 9;
    
    for(int i = 1; i<=num; i++) {
        char buf[128] = {0};
        sprintf(buf, "left%02d.jpg", i);        
        left_names.push_back(path + string(buf));
        sprintf(buf, "right%02d.jpg", i);
        right_names.push_back(path + string(buf));
    }

    
    // 1. read images
    vector<Mat> left_images;
    vector<Mat> right_images;
    for(int i = 0; i<num; i++) {
        
        Mat im = cv::imread(left_names[i].c_str(), 0);
        cout<<left_names[i]<<endl;
        assert(im.data);
        left_images.push_back(im);

        im = cv::imread(right_names[i].c_str(), 0);
        right_images.push_back(im);
    }

    // 2. detect corners
    for(int i = 0; i<num; i++) {
        cv::Size pattern_size(8, 6);
        vector<Point2f> corners;
        
        Mat im = left_images[i];
        bool found = cv::findChessboardCorners(im, pattern_size, 
        corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
        + CALIB_CB_FAST_CHECK);

        if (found) {            
            cv::TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS, 30, 0.1);
            cv::cornerSubPix(im, corners, cv::Size(11, 11), cv::Size(-1, -1), termcrit);
        }
    }
    // 3. estimate intrinsic and extrinsic parameters
    // 4. undistort image
    return 0;    
}