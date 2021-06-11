#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/subscriber.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

static const std::string OPENCV_WINDOW = "Image Window";
static const std::string OPENCV_WINDOW_LEFT = "Left Image";
static const std::string OPENCV_WINDOW_RIGHT = "Right Image";
static const std::string OPENCV_WINDOW_DEPTH = "Depth";

/*static const cv::Mat Q = (cv::Mat_<double>(4,4) << 1, 0, 0, -264.5759658813477,
 0, 1, 0, -195.3840007781982,
 0, 0, 0, 392.702728029089,
 0, 0, 0.002509779106299974, -0);
*/

/*static const cv::Mat Q = (cv::Mat_<double>(4,4) << 1, 0, 0, -1213.003723144531,
 0, 1, 0, -1054.623413085938,
 0, 0, 0, 2219.110549322086,
 0, 0, 0.002183540234675505, -0
);*/


/*static const cv::Mat Q = (cv::Mat_<double>(4,4) << 1, 0, 0, -602.878633324088,
 0, 1, 0, -524.169585645997,
 0, 0, 0, 1167.00271151386,
 0, 0, 0.001529052, -0
);*/

/*static const cv::Mat Q = (cv::Mat_<double>(4,4) << 1, 0, 0, -1200.347946166992,
 0, 1, 0, -1006.815574645996,
 0, 0, 0, 2065.845447104166,
 0, 0, 2.101691371515522, -0
);*/

/*static const cv::Mat Q = (cv::Mat_<double>(4,4) << 1, 0, 0, -908.1496887207031,
 0, 1, 0, -510.6627731323242,
 0, 0, 0, 2152.242959134286,
 0, 0, 2.101691371515522, -0);*/

static const cv::Mat Q = (cv::Mat_<double>(4,4) << 1, 0, 0, -1189.20751953125,
 0, 1, 0, -1089.432708740234,
 0, 0, 0, 2221.035543214083,
 0, 0, 0.005166713760231162, -0);


class stereoDisparity{

public:
	stereoDisparity():it_(nh_),
	image_left_sub_ ( nh_, "stereo/left/image_rect", 1 ),
	image_right_sub_( nh_, "stereo/right/image_rect", 1 ),	
	sync( StereoSyncPolicy(10), image_left_sub_, image_right_sub_)
	{
	sync.registerCallback(boost::bind(&stereoDisparity::callback, this, _1, _2));	
	cv::namedWindow(OPENCV_WINDOW, CV_WINDOW_NORMAL);
	cv::namedWindow(OPENCV_WINDOW_LEFT, CV_WINDOW_NORMAL);
	cv::namedWindow(OPENCV_WINDOW_RIGHT, CV_WINDOW_NORMAL);
	//cv::namedWindow(OPENCV_WINDOW);
	}
	~stereoDisparity(){	
	cv::destroyAllWindows();
	}
	void callback(
		const sensor_msgs::ImageConstPtr& left_msg,
		const sensor_msgs::ImageConstPtr& right_msg
){
		cv_bridge::CvImagePtr cv_ptr_left, cv_ptr_right;
		try
		{
			cv_ptr_left = cv_bridge::toCvCopy(left_msg, sensor_msgs::image_encodings::BGR8);
			cv_ptr_right = cv_bridge::toCvCopy(right_msg, sensor_msgs::image_encodings::BGR8);
		}
		catch(cv_bridge::Exception& e)
		{
			ROS_ERROR("cv_bridge exception: %s", e.what());
			return;
		}
		cv::Mat left = cv_ptr_left->image;
		cv::Mat right = cv_ptr_right->image;
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
		sgbm->setSpeckleRange(1);
		sgbm->setSpeckleWindowSize(1);
		sgbm->setDisp12MaxDiff(100);
		sgbm->setMode(cv::StereoSGBM::MODE_HH);
	 
		sgbm->compute(left, right, disparity);
	 	//sgbm->compute(right, left, disparity);
		disparity.convertTo(disparity, CV_32F, 1.0 / 16);                //divide by 16 to get the real disparity
		cv::Mat disp8U = cv::Mat(disparity.rows, disparity.cols, CV_8UC1);       //display
		
		//cv::GaussianBlur(disp8U,disp8U,cv::Size(3,3),0,0);
		//cv::medianBlur(disp8U,disp8U,3);
		
		sensor_msgs::PointCloud2 output;
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloudDepth (new pcl::PointCloud<pcl::PointXYZ>);
		cv::reprojectImageTo3D(disparity, depth, Q, true, -1);
		cvToPcl(depth, cloudDepth);
		pcl::toROSMsg(*cloudDepth, output);
		//output.header.frame_id = "depthMap";
        output.header.frame_id = "depthMap";
		//std::cout<<*cloudDepth<<std::endl;
		pcl_pub.publish(output);

		cv::normalize(disparity, disp8U, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		cv::applyColorMap(disp8U, disp8U, cv::COLORMAP_JET);
		cv::imshow(OPENCV_WINDOW_LEFT,left);
		cv::imshow(OPENCV_WINDOW_RIGHT,right);
	 	cv::imshow(OPENCV_WINDOW,disp8U);
		//cv::imshow(OPENCV_WINDOW_DEPTH,depth);
	 	cv::waitKey(1);
	}
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
private:
	ros::NodeHandle nh_;
	ros::Publisher pcl_pub = nh_.advertise<sensor_msgs::PointCloud2>("pclDepth", 1);
	image_transport::ImageTransport it_;
	typedef message_filters::Subscriber< sensor_msgs::Image > ImageSubscriber;
	
	ImageSubscriber image_left_sub_;
	ImageSubscriber image_right_sub_;

	typedef message_filters::sync_policies::ApproximateTime< sensor_msgs::Image, sensor_msgs::Image> StereoSyncPolicy;
	message_filters::Synchronizer< StereoSyncPolicy> sync;
};

int main(int argc, char** argv){
	ros::init(argc, argv, "stereo_disparity");
	stereoDisparity sD;

	while(ros::ok())
	{
		ros::spin();	
	}
	return EXIT_SUCCESS;
}
