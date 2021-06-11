#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/subscriber.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>

static const std::string OPENCV_WINDOW = "Image Window";
static const std::string OPENCV_WINDOW_LEFT = "Left Image";
static const std::string OPENCV_WINDOW_RIGHT = "Right Image";
static const std::string prefix_l = "l_";
static const std::string prefix_r = "r_";
static const std::string address_l = "/home/admini/cv_data/left/";
static const std::string address_r = "/home/admini/cv_data/right/";

int number = 1;
class stereoDisparity{

public:
	stereoDisparity():it_(nh_),
	image_left_sub_ ( nh_, "stereo/left/image_raw", 1 ),
	image_right_sub_( nh_, "stereo/right/image_raw", 1 ),	
	sync( StereoSyncPolicy(10), image_left_sub_, image_right_sub_)
	{
	sync.registerCallback(boost::bind(&stereoDisparity::callback, this, _1, _2));	
	cv::namedWindow(OPENCV_WINDOW, CV_WINDOW_NORMAL);
	//cv::namedWindow(OPENCV_WINDOW_LEFT, CV_WINDOW_AUTOSIZE);
	//cv::namedWindow(OPENCV_WINDOW_RIGHT, CV_WINDOW_AUTOSIZE);	
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
		std::stringstream ss;
		cv::Mat left = cv_ptr_left->image;
		cv::Mat right = cv_ptr_right->image;
		int k = cv::waitKey(40);
		cv::Mat h;
		cv::hconcat(left,right,h);
		//cv::imshow(OPENCV_WINDOW_LEFT,left);
		//cv::imshow(OPENCV_WINDOW_RIGHT,right);
		cv::imshow(OPENCV_WINDOW,h);
		if (k == 115){
			ss<<number;
			std::cout<<address_l+prefix_l+ss.str()+".png"<<std::endl;
			std::cout<<address_r+prefix_r+ss.str()+".png"<<std::endl;
			cv::imwrite(address_l+prefix_l+ss.str()+".png",left);
			cv::imwrite(address_r+prefix_r+ss.str()+".png",right);
			number += 1;
			}
		//std::cout<<number<<std::endl;
	}

private:
	ros::NodeHandle nh_;
	image_transport::ImageTransport it_;
	typedef message_filters::Subscriber< sensor_msgs::Image > ImageSubscriber;
	
	ImageSubscriber image_left_sub_;
	ImageSubscriber image_right_sub_;

	typedef message_filters::sync_policies::ApproximateTime< sensor_msgs::Image, sensor_msgs::Image> StereoSyncPolicy;
	message_filters::Synchronizer< StereoSyncPolicy> sync;
};

int main(int argc, char** argv){
	ros::init(argc, argv, "stereo_capture");
	stereoDisparity sD;

	while(ros::ok())
	{
		ros::spin();	
	}
	return EXIT_SUCCESS;
}
