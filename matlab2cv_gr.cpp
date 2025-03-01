#include<iostream>
#include<opencv2/calib3d.hpp>
#include<opencv2/opencv.hpp>

int main()
{	
	cv::Size imgsize;
	/*cv::Mat camMatrix_1 =(cv::Mat_<double>(3,3) << 2328.8, 4.1081, 1216.4, 0, 2325.9, 1053, 0, 0, 1 );
	cv::Mat camMatrix_2 = (cv::Mat_<double>(3,3) << 2323.1, 1.8814, 1197.7, 0, 2325.5, 1051.5, 0, 0, 1);
	cv::Mat dist_1 = (cv::Mat_<double>(1,5) << -0.069601, 0.24741, 0.00048577, 0.00030989, -0.34893);
	cv::Mat dist_2 = (cv::Mat_<double>(1,5) << -0.085466, 0.38594, 0.0007335, -0.0046183, -0.61845 );
	cv::Mat R = (cv::Mat_<double>(3,3) << 0.99997, -0.0052534, -0.0061495, 0.0052078, 0.99996, -0.0074183, 0.0061882, 0.007386, 0.99995);
	cv::Mat T = (cv::Mat_<double>(3,1) << -457.97, -1.08, 0.73792);*/

	/*cv::Mat camMatrix_1 =(cv::Mat_<double>(3,3) << 1170.2484704138, 0, 593.280170494933,0, 1164.4065970702,530.339929050228, 0, 0, 1);
	cv::Mat camMatrix_2 = (cv::Mat_<double>(3,3) << 1167.00271151386, 0, 602.878633324088,0, 1160.11492646758, 524.169585645997,0, 0, 1);
	cv::Mat dist_1 = (cv::Mat_<double>(1,5) << -0.00275883860849561, 0.045721982758491, 0.0020863316238523, -0.00758335071928252, 0);
	cv::Mat dist_2 = (cv::Mat_<double>(1,5) << -0.0777974156713806, 0.159879541273375, -0.00613611536922423, -0.00267779055986481, 0 );
	cv::Mat R = (cv::Mat_<double>(3,3) << 0.99923, 0.005414, -0.3882, -0.0066947, 0.999435574, -0.0329198, 0.0386217011, 0.033154412655, 0.9987037);
	cv::Mat T = (cv::Mat_<double>(3,1) << -607.7431, -1.342859, -1.05911);*/

    /*cv::Mat camMatrix_1 =(cv::Mat_<double>(3,3) << 2296.961069, 0, 625.398224,0, 2308.797143,501.866947, 0, 0, 1);
	cv::Mat camMatrix_2 = (cv::Mat_<double>(3,3) << 2323.375880, 0, 669.498993,0, 2333.036167, 516.090505,0, 0, 1);
    cv::Mat dist_1 = (cv::Mat_<double>(1,5) << 0.027376, -0.455105, -0.001137, -0.000992, 0);
	cv::Mat dist_2 = (cv::Mat_<double>(1,5) << -0.128655, 0.270398, 0.000259, 0.002748 , 0 );
	cv::Mat R = (cv::Mat_<double>(3,3) << 0.9998845374012215, 0.0025050295419663574, -0.014987884872093293, -0.0030114986360113027, 0.9994218681475354, -0.03386532658126264, 0.014894386254913267, 0.03390655239749609, 0.999314016194412);
	cv::Mat T = (cv::Mat_<double>(3,1) << -473.52832465906813, 3.9733771083513, 46.343077929299197);*/

    cv::Mat camMatrix_1 =(cv::Mat_<double>(3,3) << 2315.1149, 0, 1186.7538, 0, 2301.051, 1099.3925, 0, 0, 1);
	cv::Mat camMatrix_2 =(cv::Mat_<double>(3,3) << 2315.7373, 0, 1218.7981, 0, 2302.5312,1099.7659, 0, 0, 1);
	cv::Mat dist_1 = (cv::Mat_<double>(1,5) << -0.1110, 0.2800,  -0.0024, -0.0061, 0);
	cv::Mat dist_2 = (cv::Mat_<double>(1,5) << -0.1089, 0.1979,  -0.0010, -0.0042, 0);

  
	cv::Mat R = (cv::Mat_<double>(3,3) << 1, -0.0024, 0.0038, 0.0024, 1, 0.0096, -0.0038, -0.0096, 0.9999);

	cv::Mat T = (cv::Mat_<double>(3,1) << -193.5462, 0.404, -0.0224);

	cv::Mat R_1, R_2, P_1, P_2, Q;
	cv::Mat img = cv::imread("l_1.png");
	imgsize = img.size();
	
	const double alpha = 1;
	cv::stereoRectify(camMatrix_1, dist_1, camMatrix_2, dist_2, imgsize, R, T, R_1, R_2, P_1, P_2, Q,cv::CALIB_ZERO_DISPARITY,alpha, imgsize);
	std::cout<<imgsize<<std::endl;
	std::cout<< "R_1 is "<< R_1 << std::endl;
	std::cout<< "R_2 is "<< R_2 << std::endl;
	std::cout<< "P_1 is "<< P_1 << std::endl;
	std::cout<< "P_2 is "<< P_2 << std::endl;
	std::cout<< "Q is "<< Q << std::endl;
	//std::cout<< R_2 << std::endl;
	cv::imshow("img",img);
	cv::waitKey(0);
}



