#include <stdio.h>
#include <iostream> 


#include<windows.h>

#include <match/lsm.h>
#include <match/sift_matcher.h>
#include<match/sift_detector.h>

#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Jacobi>
#include <fstream>
#include <istream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>

using namespace std;
using namespace cv;


	cv::Mat image_percent_scale_8u(const cv::Mat &image, cv::Mat mask, double percent) {
		if (mask.empty()) {
			mask = cv::Mat(image.rows, image.cols, CV_8U, cv::Scalar(255));
		}

		cv::Mat mat;

		double min_max[2];
		cv::minMaxIdx(image, &min_max[0], &min_max[1], NULL, NULL, mask);
		if (percent != 0) {
			// 计算 histogram，并且计算 cdf，然后判断 clip 的最大和最小值
			std::vector<cv::Mat> mats = { image };
			int dims = 2;
			int nimage = 1;
			std::vector<int> hist_size = { 256 };
			std::vector<int> channels = { 0 };
			float step = (min_max[1] - min_max[0]) / hist_size[0];
			std::vector<float> ranges = { (float)min_max[0], (float)min_max[1] + step * 0.5f };

			cv::Mat hists;
			cv::calcHist(mats, channels, mask, hists, hist_size, ranges);
			std::vector<float> hist = hists;

			double total = cv::countNonZero(mask);
			std::vector<double> cdf(hist.size());
			int i_min_max[2] = { 0, hist_size[0] - 1 };
			for (int i = 0; i < hist.size(); ++i) {
				cdf[i] = hist[i] / total;
				if (i != 0) {
					cdf[i] += cdf[i - 1];
				}
				if (cdf[i] * 100.0 > percent && i_min_max[0] == 0) {
					i_min_max[0] = i;
				}
				if (cdf[i] * 100.0 > 100 - percent && i_min_max[1] == hist_size[0] - 1) {
					i_min_max[1] = i;
				}
			}
			min_max[0] = ranges[0] + step * i_min_max[0];
			min_max[1] = ranges[0] + step * i_min_max[1];
		}

		image.convertTo(mat, CV_32F);
		mat = (mat - min_max[0]) / (min_max[1] - min_max[0]) * 254.0f + 1.0f;
		mat.setTo(cv::Scalar(1.0f), mat < 1.0f);
		mat.setTo(cv::Scalar(255.0f), mat > 255.0f);
		mat.convertTo(mat, CV_8U);
		mat.setTo(cv::Scalar(0), mask == 0);

		return mat;
	}

	void readme()
	{
		printf("This project takes input from 3 video streams and stitches the videos, frame by frame.");
	}


	Mat calculate_h_matrix(Mat image1, Mat image2, Mat gray_image1, Mat gray_image2)
	{
		
		//-- Step 1: Detect the keypoints using SIFT Detector

		//-- Step 2: Calculate descriptors (feature vectors)

		std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;
		cv::Mat descriptors_object, descriptors_scene;


		cv::Mat mask = cv::Mat(image1.rows, image1.cols, CV_8U, cv::Scalar(255));
		double percent = 2.0;
		cv::Mat mat1 = image_percent_scale_8u(gray_image1, mask, percent);
		cv::Mat mat2 = image_percent_scale_8u(gray_image2, mask, percent);

		//std::shared_ptr<h2o::SiftDetector> sift_;
		h2o::SiftDetector sift;
		std::tie(keypoints_object, descriptors_object) = sift.detect_and_compute(mat1);
		std::tie(keypoints_scene, descriptors_scene) = sift.detect_and_compute(mat2);

		//std::tie(keypoints_object, descriptors_object) = sift_->detect_and_compute(mat1);
		//std::tie(keypoints_scene, descriptors_scene) = sift_->detect_and_compute(mat2);
				
		//-- Step 3: Matching descriptor vectors using FLANN matcher
		// match sift
		h2o::SiftMatcherParam sift_param;
		h2o::SiftMatcher sift_matcher;
		sift_matcher.set_match_param(sift_param);
		sift_matcher.set_train_data(keypoints_object, descriptors_object);
		std::vector<cv::DMatch> matches = sift_matcher.match(keypoints_scene, descriptors_scene); 
		//sift_matcher.draw_matches(image1, image2, &keypoints_scene, &keypoints_object, matches);
		for (int i = 0; i < matches.size(); i++) {
			cv::circle(image1, cv::Point(keypoints_object.at(matches[i].trainIdx).pt), 1, 1, 1);
			cv::circle(image2, cv::Point(keypoints_scene.at(matches[i].queryIdx).pt), 1, 1, 1);
		}
		auto good_matches = matches;

		//FlannBasedMatcher matcher;
		//std::vector< DMatch > matches;
		//matcher.match(descriptors_object, descriptors_scene, matches);
		////matcher.match(descriptors_object, descriptors_scene, matches);

		//double max_dist = 0; double min_dist = 100;

		////-- Quick calculation of max and min distances between keypoints 
		//for (int i = 0; i < descriptors_object.rows; i++)
		//{
		//	double dist = matches[i].distance;
		//	if (dist < min_dist) min_dist = dist;
		//	if (dist > max_dist) max_dist = dist;
		//}

		//printf("-- Max dist: %f \n", max_dist);
		//printf("-- Min dist: %f \n", min_dist);


		////-- Use only "good" matches (i.e. whose distance is less than 3*min_dist )
		//std::vector< DMatch > good_matches;
		//cv::Mat result;
		//// cv::Mat result23;
		//
		//// cv::Mat H23;
		//for (int i = 0; i < descriptors_object.rows; i++)
		//{
		//	if (matches[i].distance < 3 * min_dist)
		//	{
		//		good_matches.push_back(matches[i]);
		//	}
		//}



		//calculate h matrix
		cv::Mat H;
		std::vector< Point2f > obj;
		std::vector< Point2f > scene;

		for (int i = 0; i < good_matches.size(); i++)
		{
			//-- Get the keypoints from the good matches
			obj.push_back(keypoints_object[good_matches[i].trainIdx].pt);
			scene.push_back(keypoints_scene[good_matches[i].queryIdx].pt);
		}


		// Find the Homography Matrix for img 1 and img2
		H = findHomography(obj, scene);
		return H;
	}


	Mat stitch_image(Mat image1, Mat image2, Mat H)
	{

		cv::Mat result;
		// cv::Mat result23;
		warpPerspective(image1, result, H, cv::Size(image1.cols + image2.cols, image1.rows));
		cv::Mat half(result, cv::Rect(0, 0, image2.cols, image2.rows));
		image2.copyTo(half);


		// cv::resize(result,result, Size(image1.cols,image1.rows),INTER_LINEAR);

		// cv::imshow("Result", result);

		// Mat ycrcb;

	 //    cvtColor(result,ycrcb,CV_BGR2YCrCb);

	 //    vector<Mat> channels;
	 //    split(ycrcb,channels);

	 //    equalizeHist(channels[0], channels[0]);

	 //    Mat dst;
	 //    merge(channels,ycrcb);

	 //    cvtColor(ycrcb,dst,CV_YCrCb2BGR);

		// cv::imshow("Hist_Equalized_Result", dst);
		// cv::resize(dst,dst,image1.size());
		// cv::imwrite("./Result/Result.jpg", dst);
		// // cv::imwrite("./data/cam_left.jpg", image1);
		// // cv::imwrite("./data/cam_right.jpg", image2);
		// waitKey(0);
		return result;
	}


	// Mat hist_equalization()
	// {
	// cvtColor(img2, img_gray2, COLOR_BGR2GRAY);
	// Mat ycrcb;
	// cvtColor(result,ycrcb,CV_BGR2YCrCb);
	// vector<Mat> channels;
	// split(ycrcb,channels);
	// equalizeHist(channels[0], channels[0]);
	// Mat dst;
	// merge(channels,ycrcb);
	// cvtColor(ycrcb,img,CV_YCrCb2BGR);

	// }

	int main(int argc, char** argv)
	{
		readme();
		// if( argc != 4 )
		// { readme(); return -1; }
		Mat gray_image1;
		Mat gray_image2;
		Mat gray_image3;
		Mat gray_image4;
		Mat img, img2;
		Mat gray_img;
		Mat result;
		Mat img_gray, img_gray2, img_gray3, img_gray4, img_gray5;
		Mat img3, img4, img5, img6;

		// Load the images
		VideoCapture cap1(1);
		VideoCapture cap2(2);
		VideoCapture cap3(3);


		for (;;)
		{

			Mat image1=cv::imread("F:\\wzd\\thesis\\Image-Stitching\\data\\S1.jpg");
			//cap1 >> image1; // get a new frame from camera
			cvtColor(image1, gray_image1, COLOR_BGR2GRAY);

			Mat image2 = cv::imread("F:\\wzd\\thesis\\Image-Stitching\\data\\S2.jpg");;
			//cap2 >> image2; // get a new frame from camera
			cvtColor(image2, gray_image2, COLOR_BGR2GRAY);

			Mat image3 = cv::imread("F:\\wzd\\thesis\\Image-Stitching\\data\\S3.jpg");;
			//cap3 >> image3; // get a new frame from camera
			cvtColor(image3, gray_image3, COLOR_BGR2GRAY);

			imshow("first image", image1);
			cv::imwrite("./data/Image1.jpg", image1);

			imshow("second image", image2);
			cv::imwrite("./data/Image2.jpg", image2);

			imshow("third image", image3);
			cv::imwrite("./data/Image3.jpg", image3);

			if (!gray_image1.data || !gray_image2.data)
			{
				std::cout << " --(!) Error reading images " << std::endl; return -1;
			}




			// Mat image1 = imread("../Image_stitching/data/S1.jpg");
			// cvtColor(image1, gray_image1, COLOR_BGR2GRAY);

			// Mat image2 =imread("../Image_stitching/data/S2.jpg");
			// cvtColor(image2, gray_image2, COLOR_BGR2GRAY);

			// Mat image3 = imread("../Image_stitching/data/S3.jpg");
			// cvtColor(image3, gray_image3, COLOR_BGR2GRAY);

			// Mat image4 = imread("../Image_stitching/data/S5.jpg");
			// cvtColor(image4, gray_image4, COLOR_BGR2GRAY);

			// imshow("first image",image1);
			// imshow("second image",image2);
			// imshow("third image",image3);
			// imshow("fourth image", image4);

			// if( !gray_image1.data || !gray_image2.data )
			// 	{ std::cout<< " --(!) Error reading images " << std::endl; return -1; }




			Mat H12 = calculate_h_matrix(image2, image1, gray_image2, gray_image1);
			// Mat H13 = calculate_h_matrix(image1,image3, gray_image1, gray_image3);
			Mat H23 = calculate_h_matrix(image3, image2, gray_image3, gray_image2);
			// Mat H34 = calculate_h_matrix(image3,image4, gray_image3, gray_image4);



			/*The main logic is to chose a central image which is common to the 3 images.
			In this code, there are 3 images, in the order: Image1, Image2 and Image 3 respectively.
			The Image2 is thus comman to all three image and thus we chose this as the central image
			and calculate the homography matrices of other images with respect to this image*/

			//Stitch Image 2 and Image 3 and saved in img
			img = stitch_image(image3, image2, H23);
			cvtColor(img, img_gray, COLOR_BGR2GRAY);

			// //Finding the largest contour i.e remove the black region from image
			threshold(img_gray, img_gray, 25, 255, THRESH_BINARY); //Threshold the gray
			vector<vector<Point> > contours; // Vector for storing contour
			vector<Vec4i> hierarchy;
			//cv::findContours(img_gray, contours, hierarchy, 0); // Find the contours in the image
			cv::findContours(img_gray, contours, 0, 0); // Find the contours in the image
			int largest_area = 0;
			int largest_contour_index = 0;
			Rect bounding_rect;

			for (int i = 0; i < contours.size(); i++) // iterate through each contour. 
			{
				double a = contourArea(contours[i], false);  //  Find the area of contour
				if (a > largest_area) {
					largest_area = a;
					largest_contour_index = i;                //Store the index of largest contour
					bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
				}

			}

			// Scalar color( 255,255,255);
			//img = img(Rect(bounding_rect.x, bounding_rect.y, bounding_rect.width, bounding_rect.height));





			//Stitch Image 1 and Image 2 and saved in img2
			img2 = stitch_image(image2, image1, H12);
			cvtColor(img2, img_gray2, COLOR_BGR2GRAY);

			//Finding the largest contour i.e remove the black region from image
			threshold(img_gray2, img_gray2, 25, 255, THRESH_BINARY); //Threshold the gray
			vector<vector<Point> > contours2; // Vector for storing contour
			vector<Vec4i> hierarchy2;
			findContours(img_gray2, contours2, 0,0); // Find the contours in the image
			int largest_area2 = 0;
			int largest_contour_index2 = 0;
			Rect bounding_rect2;

			for (int i = 0; i < contours2.size(); i++) // iterate through each contour. 
			{
				double a = contourArea(contours2[i], false);  //  Find the area of contour
				if (a > largest_area2) {
					largest_area2 = a;
					largest_contour_index2 = i;                //Store the index of largest contour
					bounding_rect2 = boundingRect(contours2[i]); // Find the bounding rectangle for biggest contour
				}

			}

			img2 = img2(Rect(bounding_rect2.x, bounding_rect2.y, bounding_rect2.width, bounding_rect2.height));

			// //Stitch Image 3 and Image 4 and saved in img3
			// img3 = stitch_image(image3, image4, H34);
			// img3 = img3(Rect(0,0,(img3.cols/2),(img3.rows)));
			// cvtColor(img3, img_gray3, COLOR_BGR2GRAY);


			//Show img
			cv::imshow("Hist_Equalized_Result of Image 2 and Image 3", img);
			cv::imwrite("./Result/Result23.jpg", img);
			// waitKey(0);

			//Show img2
			cv::imshow("Hist_Equalized_Result of Image 1 and Image 2", img2);
			cv::imwrite("./Result/Result12.jpg", img2);
			// waitKey(0);

			// //Show img3
			// // cv::resize(img2,img2,image1.size());
			// cv::imshow("Hist_Equalized_Result of Image 3 and Image 4", img3);
			// cv::imwrite("./Result/Result34.jpg", img3);
			// waitKey(0);


			// Stitch (Image 1 and Image 2) and (Image 2 and Image 3)
			Mat H123 = calculate_h_matrix(img, img2, img_gray, img_gray2);
			img4 = stitch_image(img, img2, H123);
			// img4 = img4(Rect(0,0,(img4.cols*7/10),(img4.rows)));
			// cvtColor(img4, img_gray4, COLOR_BGR2GRAY);

			// //Stitch (Image 2 and Image 3) and (Image 3 and Image 4)
			// Mat H234 = calculate_h_matrix(img3,img, img_gray3, img_gray);
			// img5 = stitch_image(img3,img, H234);
			// img5 = img5(Rect(0,0,(img5.cols*3/4),(img5.rows)));
			// cvtColor(img5, img_gray5, COLOR_BGR2GRAY);

			// //Stitch (Image 1 and Image 2 and Image 3) and (Image 2 and Image 3 and Image 4)
			// Mat H1234 = calculate_h_matrix(img5,img4, img_gray5, img_gray4);
			// img6 = stitch_image(img5,img4, H1234);

			cv::imshow("Hist_Equalized_Result of Image 1 and Image 2 and Image 3", img4);
			cv::imwrite("./Result/Result123.jpg", img4);
			// waitKey(0);

			// cv::imshow("Hist_Equalized_Result of Image 2 and Image 3 and Image 4" , img5);
			// cv::imwrite("./Result/Result234.jpg", img5);
			// waitKey(0);

			// cv::imshow("Hist_Equalized_Result of Image 1 and Image 2 and Image 3 and Image 4" , img6);
			// cv::imwrite("./Result/Result1234.jpg", img6);
			// waitKey(0);

		}

	}

