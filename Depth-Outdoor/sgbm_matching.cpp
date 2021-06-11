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

using namespace std;
using namespace cv;

 
int main()
{
	Mat left = imread("/home/admini/Normilization/l_5.png", IMREAD_GRAYSCALE);
	Mat right = imread("/home/admini/Normilization/r_5.png", IMREAD_GRAYSCALE);
	Mat disp;
 
	int mindisparity = 0;
	int ndisparities = 256;  
	int SADWindowSize = 3; 
 
	//SGBM
	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(mindisparity, ndisparities, SADWindowSize);
	int P1 = 8 * left.channels() * SADWindowSize* SADWindowSize;
	int P2 = 32 * left.channels() * SADWindowSize* SADWindowSize;
	sgbm->setP1(P1);
	sgbm->setP2(P2);
 
	sgbm->setPreFilterCap(15);
	sgbm->setUniquenessRatio(15);
	sgbm->setSpeckleRange(2);
	sgbm->setSpeckleWindowSize(120);
	sgbm->setDisp12MaxDiff(1);
	sgbm->setMode(cv::StereoSGBM::MODE_HH);
 
	sgbm->compute(left, right, disp);
 
	disp.convertTo(disp, CV_32F, 1.0 / 16);                //除以16得到真实视差值
	Mat disp8U = Mat(disp.rows, disp.cols, CV_8UC1);       //显示
	normalize(disp, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);
	namedWindow("disparity", WINDOW_NORMAL); 	
	imshow("disparity",disp8U);
 	waitKey(0);
	imwrite("results_SGBM.jpg", disp8U);
	return 0;
}
