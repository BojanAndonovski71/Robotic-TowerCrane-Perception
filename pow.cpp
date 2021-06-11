#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;
using namespace std;

int main()
{
    // Load the image as grayscale
    cv::Mat1b src = cv::imread("/home/admini/Normilization/test1.jpg", cv::IMREAD_GRAYSCALE);

    // Convert to double for "pow"
    cv::Mat1d dsrc;
    src.convertTo(dsrc, CV_64F);

    // Compute the "pow"
    cv::Mat1d ddst;
    cv::pow(dsrc, 0.91, ddst);

    // Convert back to uchar
    cv::Mat1b dst;
    ddst.convertTo(dst, CV_8U);

    // Show results
    imshow("SRC", src);

  	double param, result;
  	param = 0.5;
  	result = tgamma (param);
  	printf ("tgamma(%f) = %f\n", param, result );
    imshow("DST", dst);
    cv::waitKey(0);

    return 0;
}
