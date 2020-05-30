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

cv::Mat drawmatch(cv::Mat mat_ground, cv::Mat mat_aerial, std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> keys_render, std::vector<cv::KeyPoint> keys_ground, cv::Scalar &color = cv::Scalar(255, 0, 0, 0), int thickness = 3)
{

	int crosssize = 180;
	auto mat_ground_size = mat_ground.size();

	if (mat_ground.type() != mat_aerial.type())
	{
		cv::cvtColor(mat_ground, mat_ground, cv::COLOR_RGBA2RGB);
	}

	if (mat_ground.size != mat_aerial.size)
		cv::resize(mat_ground, mat_ground, cv::Size(mat_aerial.cols, mat_aerial.rows), 0, 0);

	cv::Mat ter_aer_mat;
	ter_aer_mat.create(std::max<int>(mat_ground.rows, mat_ground.rows), mat_ground.cols + mat_ground.cols,
		mat_aerial.type()); // des.create()
	cv::Mat r1 = ter_aer_mat(cv::Rect(0, 0, mat_ground.cols, mat_ground.rows));
	mat_ground.copyTo(r1);
	cv::Mat r2 = ter_aer_mat(cv::Rect(mat_ground.cols, 0, mat_ground.cols, mat_ground.rows));
	mat_aerial.copyTo(r2);

	cv::Mat ter_aer_mat_vertical;
	ter_aer_mat_vertical.create(mat_aerial.rows + mat_aerial.rows, mat_aerial.cols,
		mat_aerial.type()); // des.create()
	cv::Mat r11 = ter_aer_mat_vertical(cv::Rect(0, 0, mat_aerial.cols, mat_aerial.rows));
	mat_ground.copyTo(r11);
	cv::Mat r22 = ter_aer_mat_vertical(cv::Rect(0, mat_aerial.rows, mat_aerial.cols, mat_aerial.rows));
	mat_aerial.copyTo(r22);

	int countKey = 0;
	for (const auto &match : matches) {


		// cv::line(mat_aerial, cv::Point(key_aerial[i].x() - crosssize / 2, key_aerial[i].y()),
		//         cv::Point(key_aerial[i].x() + crosssize / 2, key_aerial[i].y()), color, thickness, 8, 0);
		////绘制竖线
		// cv::line(mat_aerial, cv::Point(key_aerial[i].x(), key_aerial[i].y() - crosssize / 2),
		//         cv::Point(key_aerial[i].x(), key_aerial[i].y() + crosssize / 2), color, thickness, 8, 0);

		line(ter_aer_mat,
			cv::Point(keys_render[match.queryIdx].pt.x * mat_aerial.cols / mat_ground_size.width,
				keys_render[match.queryIdx].pt.y * mat_aerial.rows / mat_ground_size.height),
			cv::Point(keys_ground[match.trainIdx].pt.x + mat_aerial.cols, keys_ground[match.trainIdx].pt.y), color, thickness);

		line(ter_aer_mat_vertical,
			cv::Point(keys_render[match.queryIdx].pt.x * mat_aerial.cols / mat_ground_size.width,
				keys_render[match.queryIdx].pt.y * mat_aerial.rows / mat_ground_size.height),
			cv::Point(keys_ground[match.trainIdx].pt.x, keys_ground[match.trainIdx].pt.y + mat_aerial.rows), color, thickness);

		/*cv::arrowedLine(mat_ground, cv::Point(keys_render[match.queryIdx].pt.x * mat_aerial.cols / mat_ground_size.width,
			keys_render[match.queryIdx].pt.y * mat_aerial.rows / mat_ground_size.height),
			cv::Point(keys_ground[match.trainIdx].pt.x, keys_ground[match.trainIdx].pt.y + mat_aerial.rows),  cv::Scalar(255, 0, 255), 3, 8, 0, 0.3);*/

		/*  circle(mat_ground,
				 cv::Point(key_ground.x() * mat_aerial.cols / mat_ground_size.width,
						   key_ground.y() * mat_aerial.rows / mat_ground_size.height),
				 5, color, 5);
		  circle(mat_aerial, cv::Point(key_aerial[i].x(), key_aerial[i].y()), 2, color, 3);*/

	}
	return ter_aer_mat;
}



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

		h2o::SiftDetector sift;
		std::tie(keypoints_object, descriptors_object) = sift.detect_and_compute(mat1);
		std::tie(keypoints_scene, descriptors_scene) = sift.detect_and_compute(mat2);

		//-- Step 3: Matching descriptor vectors using FLANN matcher
		// match sift
		h2o::SiftMatcherParam sift_param;
		h2o::SiftMatcher sift_matcher;
		sift_matcher.set_match_param(sift_param);
		sift_matcher.set_train_data(keypoints_object, descriptors_object);
		std::vector<cv::DMatch> matches = sift_matcher.match(keypoints_scene, descriptors_scene); 
		/*for (int i = 0; i < matches.size(); i++) {
			cv::circle(image1, cv::Point(keypoints_object.at(matches[i].trainIdx).pt), 5, cv::Scalar(0, 255, 0, 0), 5);
			cv::circle(image2, cv::Point(keypoints_scene.at(matches[i].queryIdx).pt), 5, cv::Scalar(0, 255, 0, 0), 5);
		}*/
		auto good_matches = matches;

		Mat match_mat = drawmatch(image2, image1, matches, keypoints_scene, keypoints_object);

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

		return result;
	}


	int main(int argc, char** argv)
	{
		Mat gray_image1;
		Mat gray_image2;
		Mat gray_image3;
		Mat gray_image4;
		Mat img, img2;
		Mat gray_img;
		Mat result;
		Mat img_gray, img_gray2, img_gray3, img_gray4, img_gray5;
		Mat img3, img4, img5, img6;

		Mat image1 = cv::imread("D:\\SWJTUFiles\\thesis\\ImageStitching\\data\\temp\\cam2.png");
		cvtColor(image1, gray_image1, COLOR_BGR2GRAY);

		Mat image2 = cv::imread("D:\\SWJTUFiles\\thesis\\ImageStitching\\data\\temp\\cam3.png");;
		cvtColor(image2, gray_image2, COLOR_BGR2GRAY);

		Mat image3 = cv::imread("D:\\SWJTUFiles\\thesis\\ImageStitching\\data\\temp\\cam4.png");;
		cvtColor(image3, gray_image3, COLOR_BGR2GRAY);

		if (!gray_image1.data || !gray_image2.data)
		{
			std::cout << " --(!) Error reading images " << std::endl; return -1;
		}


		Mat H23 = calculate_h_matrix(image3, image2, gray_image3, gray_image2);

		Mat H12 = calculate_h_matrix(image2, image1, gray_image2, gray_image1);
		// Mat H13 = calculate_h_matrix(image1,image3, gray_image1, gray_image3);
		// Mat H34 = calculate_h_matrix(image3,image4, gray_image3, gray_image4);

		/*The main logic is to chose a central image which is common to the 3 images.
		In this code, there are 3 images, in the order: Image1, Image2 and Image 3 respectively.
		The Image2 is thus comman to all three image and thus we chose this as the central image
		and calculate the homography matrices of other images with respect to this image*/

		//Stitch Image 2 and Image 3 and saved in img
		img = stitch_image(image3, image2, H23);

		img2 = stitch_image(image2, image1, H12);
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
		findContours(img_gray2, contours2, 0, 0); // Find the contours in the image
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

	}

