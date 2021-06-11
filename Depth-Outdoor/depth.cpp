#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>
//#include <pcl_conversions/pcl_conversions.h>

//static const std::string OPENCV_WINDOW = "Image Window";
//static const std::string OPENCV_WINDOW_LEFT = "Left Image";
//static const std::string OPENCV_WINDOW_RIGHT = "Right Image";
//static const std::string OPENCV_WINDOW_DEPTH = "Depth";


using namespace std;
using namespace cv;


	void gammaCorrection(const cv::Mat &src, const cv::Mat &dst, const double gamma_)
	
    {
    CV_Assert(gamma_ >= 0);
    //! [changing-contrast-brightness-gamma-correction]
    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for( int i = 0; i < 256; ++i)
        p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma_) * 255.0);

    cv::LUT(src, lookUpTable, dst);
    //! [changing-contrast-brightness-gamma-correction]
	}

	void imageShift(const cv::Mat &src, const cv::Mat &dst, const double x_offset, const double y_offset)
	
	{
	cv::Mat trans_mat = (cv::Mat_<double>(2,3)<<1, 0 ,x_offset, 0, 1,y_offset);
	cv::warpAffine(src, dst, trans_mat, src.size());
	}

	void cvToPcl(const cv::Mat &src, const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz)
	{

    for (int rows = 0; rows < src.rows; ++rows) {
        for (int cols = 0; cols < src.cols; ++cols) {
            cv::Point3f point = src.at<cv::Point3f>(rows, cols);
            pcl::PointXYZ pcl_point(point.x, point.y, point.z);
            cloud_xyz->push_back(pcl_point);
        											}
    											}
	}


int main(int argc, char** argv)
{


	static const cv::Mat Q = (cv::Mat_<double>(4,4) << 1, 0, 0, -1213.003723144531,
 	0, 1, 0, -1054.623413085938,
	0, 0, 0, 2219.110549322086,
 	0, 0, 0.002183540234675505, -0
	);
//class stereoDisparity{

//public:

		cv::Mat left = imread("/home/admini/Normilization/l_5.png", IMREAD_GRAYSCALE);
		cv::Mat right = imread("/home/admini/Normilization/r_5.png", IMREAD_GRAYSCALE);
		cv::Mat disparity;
		cv::Mat_<cv::Vec3f> depth;

		int mindisparity = 0;
		int ndisparities = 256;
		int SADWindowSize = 3;

		//cv::medianBlur(left,left,5);
		//cv::medianBlur(right,right,5);
		//GaussianBlur(left,left,Size(3,3),1,1);
		//GaussianBlur(right,right,Size(3,3),1,1);
		gammaCorrection(left,left,1);
		gammaCorrection(right, right, 1);
		//imageShift(right,right,0,7);
		//imageShift(left,left,0,-7);

		//SGBM
		cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(mindisparity, ndisparities, SADWindowSize);
		int P1 = 8 * left.channels() * SADWindowSize* SADWindowSize;
		int P2 = 64 * right.channels() * SADWindowSize* SADWindowSize;
		sgbm->setP1(P1);
		sgbm->setP2(P2);

		sgbm->setPreFilterCap(1);
		sgbm->setUniquenessRatio(1);
		sgbm->setSpeckleRange(1);  //2
		sgbm->setSpeckleWindowSize(1);//100
		sgbm->setDisp12MaxDiff(100);
		sgbm->setMode(cv::StereoSGBM::MODE_HH);

		sgbm->compute(left, right, disparity);
	 	//sgbm->compute(right, left, disparity);
		disparity.convertTo(disparity, CV_32F, 1.0 / 16);                //divide by 16 to get the real disparity
		cv::Mat disp8U = cv::Mat(disparity.rows, disparity.cols, CV_8UC1);       //display

		//cv::GaussianBlur(disp8U,disp8U,cv::Size(3,3),0,0);
		//cv::medianBlur(disp8U,disp8U,3);

		//sensor_msgs::PointCloud2 output;
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloudDepth (new pcl::PointCloud<pcl::PointXYZ>);
		cv::reprojectImageTo3D(disparity, depth, Q, true, -1);
		cvToPcl(depth, cloudDepth);
		//pcl::toROSMsg(*cloudDepth, output);
		//output.header.frame_id = "depthMap";
		std::cout<<*cloudDepth<<std::endl;
		//pcl_pub.publish(output);

		cv::normalize(disparity, disp8U, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		cv::applyColorMap(disp8U, disp8U, cv::COLORMAP_JET);
        namedWindow("left", WINDOW_NORMAL);
		cv::imshow("left",left);
        namedWindow("right", WINDOW_NORMAL);
		cv::imshow("right",right);
        namedWindow("disparity", WINDOW_NORMAL);
	 	cv::imshow("disparity",disp8U);
        //cv::imwrite("disparity",disp8U);
        //namedWindow("depth", WINDOW_NORMAL);
		//cv::imshow("depth",depth);
	 	//cv::waitKey(0);

		pcl::visualization::CloudViewer viewer ("Cloud_filtered");
  		viewer.showCloud(cloudDepth);
  		while(!viewer.wasStopped ())
  		{
  		}
       waitKey(0);  
       //imwrite("/home/admini/Depth-Outdoor/test.jpg", disp8U);                                        // Wait for a keystroke in the window
       return 0;
	}






