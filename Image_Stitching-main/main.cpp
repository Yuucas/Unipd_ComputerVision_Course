#include <iostream>
#include <memory>
#include <vector>
#include "opencv2/opencv.hpp" 
#include <opencv2/core.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "PanoramicUtils.h"

//PARAMETERS
const double ratio = 6;
std::vector<cv::KeyPoint> keypoints;
cv::Mat descriptors;
std::vector<cv::DMatch> matches;


void PanoramicUtil(std::vector<cv::Mat>& images){

	for (int i = 0; i < images.size(); i++) {
		const double  FoV = 66.0;  //The FoV of the camera
		const double angle = FoV / 2;
		PanoramicUtils cylindrical;
		images[i] = cylindrical.cylindricalProj(images[i], angle); //Convert the images cylindrical surface

	}
	cv::imshow("Cylindrical", images[0]);
	cv::waitKey(0);
}


void detect_and_compute(std::vector<cv::Mat>& images, std::vector<std::vector<cv::KeyPoint>>& total_keypoints, std::vector<cv::Mat>& total_descriptors) {

	for (int i = 0; i < images.size(); i++) {

		cv::Ptr<cv::ORB> orb = cv::ORB::create(2000); //Define the ORB detector
		orb->detectAndCompute(images[i], cv::Mat(), keypoints, descriptors); //Extract the ORB features from the images
		total_keypoints.push_back(keypoints);  //Put together keypoints of the images
		total_descriptors.push_back(descriptors);  //Put together descriptors of the images
		
		cv::Mat drawn_keypoints;

		//Draw keypoints on the images and flags are setting drawing features
		cv::drawKeypoints(images[i], keypoints, drawn_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

		cv::imshow("Keypoints", drawn_keypoints);
		cv::waitKey(0);
		keypoints.clear();

	}
}


void match(std::vector<cv::Mat>& images, std::vector<cv::Mat>& total_descriptors, 
	std::vector<std::vector<cv::DMatch>>& total_matches) {

	//Matching descriptor vectors using BFMatcher
	//Compute the matches between the different features of images
	cv::Ptr<cv::BFMatcher> orb = cv::BFMatcher::create(cv::NORM_HAMMING, false);

	for (int i = 1; i < images.size(); i++)
	{
		//Compute the image with the next image
		orb->match(total_descriptors[i - 1], total_descriptors[i], matches);
		total_matches.push_back(matches);
	}

	matches.clear(); 

}


void RefineMatches(std::vector<std::vector<cv::DMatch>>& total_matches, 
	std::vector<std::vector<cv::DMatch>>& total_RefinedMatches, double ratio) {

	//Refine the matches

	float distance;
	float min_distance;
	
	std::vector<cv::DMatch> refined_matches;
	std::vector<float> minDist_matches; //Contains the minimum distance between the matches of images.

	//Compute the minimum distance between the matches
	for (int i = 0; i < total_matches.size(); i++)
	{
		min_distance = total_matches[i][0].distance; //The minimum distance found among the matches. 

		std::cout << "Min Distance\n" << min_distance << std::endl;

		for (int j = 0; j < total_matches[i].size(); j++)
		{
			distance = total_matches[i][j].distance;

			if (distance < min_distance) 
			{
				min_distance = distance;  //Specify the certain distance

			}
		}

		minDist_matches.push_back(min_distance); //Push the minimum distances to matches

	}

	for (int i = 0; i < total_matches.size(); i++) {

		for (int j = 0; j < total_matches[i].size(); j++) {

			distance = total_matches[i][j].distance;

			//Only keep the matches If minimum distance for each matches is larger than others
			//Ratio is used to find convenient distance
			if (distance <= minDist_matches[i] * ratio) {  

				refined_matches.push_back(total_matches[i][j]);

			}
		}

		total_RefinedMatches.push_back(refined_matches); //put together
		refined_matches.clear(); 

	}

}


void findGoodMatches(std::vector<std::vector<cv::KeyPoint>>& total_keypoints,
	std::vector<std::vector<cv::DMatch>>& total_RefinedMatches, 
	std::vector<std::vector<cv::DMatch>>& total_goodMatches) {

	std::vector<cv::DMatch> good_matches; //Used to store good matches from refined matches

	//Parameter for Homograph
	std::vector<cv::Point2f> object; //2D points
	std::vector<cv::Point2f> scene;
	cv::Mat match_mask;   //Used to seperate a part of the matches
	std::vector<cv::Mat> total_matchMask;

	for (int i = 0; i < total_keypoints.size() - 1; i++) {

		for (int j = 0; j < static_cast<int>(total_RefinedMatches[i].size()); j++) {

			//queryIdx is the index of the descriptor in the list of query descriptors
			object.push_back(total_keypoints[i][total_RefinedMatches[i][j].queryIdx].pt);  
			//trainIdx is the index of the descriptor in the list of train descriptors
			scene.push_back(total_keypoints[i + 1][total_RefinedMatches[i][j].trainIdx].pt);
		}

		//Find the Homography Matrix for refined matches
		findHomography(object, scene, cv::RANSAC, 3, match_mask);
		total_matchMask.push_back(match_mask);  

		object.clear(); 
		scene.clear(); 
	}

	//Find the total good matches
	for (int i = 0; i < total_RefinedMatches.size(); i++) {

		for (int j = 0; j < total_RefinedMatches[i].size(); j++) {

			if (total_matchMask[i].Mat::at<char>(j, 0)) { //Return a reference to the specified match mask array 

				good_matches.push_back(total_RefinedMatches[i][j]);
			}
		}

		total_goodMatches.push_back(good_matches);
		good_matches.clear(); 
	}
}


void findDistance(std::vector<cv::Mat>& images, std::vector< std::vector<float>>& total_distance,
	std::vector< std::vector<cv::KeyPoint>>& total_keypoints, std::vector< std::vector<cv::DMatch>>& total_goodMatches) {

	cv::Point2f object; //2D points
	cv::Point2f scene;
	float dist;
	std::vector<float> dist_vec;

	for (int i = 0; i < total_keypoints.size() - 1; i++)
	{
		for (int j = 0; j < total_goodMatches[i].size(); j++)
		{
			//Get the keypoints from total good matches
			object = total_keypoints[i][total_goodMatches[i][j].queryIdx].pt; //Coordinates of the keypoints
			scene = total_keypoints[i + 1][total_goodMatches[i][j].trainIdx].pt;
			dist = images[i].cols - object.x + scene.x;
			dist_vec.push_back(dist);
		}
		total_distance.push_back(dist_vec);
		dist_vec.clear();
	}
}

void merge(std::vector<cv::Mat>& images, cv::Mat& panoramic, std::vector<float>& total_MeanDistance, 
	std::vector<std::vector<float>>& total_distance, std::vector<std::vector<cv::DMatch>>& total_goodMatches) {

	//Compute the mean distance between the images
	float dist = 0;;
	cv::Mat ref_mat;
	panoramic = images[0];

	//Find the total mean distance
	for (int i = 0; i < total_goodMatches.size(); i++) {

		for (int j = 0; j < total_goodMatches[i].size(); j++) {

			dist = dist + total_distance[i][j];

		}

		dist = dist / total_goodMatches[i].size();
		total_MeanDistance.push_back(dist);

	}

	//Compute Translation 
	cv::Mat ref_mat2(2, 3, CV_64F, cv::Scalar(0));  //Must be 2x3 transformation matrix

	ref_mat2.cv::Mat::at<double>(0, 0) = 1;
	ref_mat2.cv::Mat::at<double>(1, 1) = 1;

	for (int i = 0; i < images.size() - 1; i++) {

		ref_mat2.cv::Mat::at<double>(0, 2) = -total_MeanDistance[i];

		//Transform the images using the specified matrix (ref_mat, ref_mat2)
		warpAffine(images[i + 1], ref_mat, ref_mat2, cv::Size(images[i + 1].cols - total_MeanDistance[i], images[i + 1].rows), 
			cv::INTER_CUBIC, cv::BORDER_DEFAULT, cv::Scalar(-1));
		cv::hconcat(panoramic, ref_mat, panoramic); //Apply horizontal concatenation

	}

}


int main(int argc, char** argv) {

	//Load images
	std::vector<cv::String> folder_vector;
	cv::String folder = "C:/Users/yukse/OneDrive/Masaüstü/Lectures/Computer Vision/LAB 4/*.bmp";
	cv::glob(folder, folder_vector, true);

	std::vector<cv::Mat> images;
	size_t count = folder_vector.size(); //number of bmp files in images folder
	for (size_t i = 0; i < count; i++) {
		images.push_back(cv::imread(folder_vector[i]));
	}


	//The function of converting the images cylindrical surface
	PanoramicUtil(images);

	//Extract the ORB features from the images
	std::vector<cv::Mat> total_descriptor;
	std::vector<std::vector<cv::KeyPoint>> total_keypoints; 
	detect_and_compute(images, total_keypoints, total_descriptor);

	//Find the nearest matches of a couple of images
	std::vector<std::vector<cv::DMatch>> total_matches; 
	match(images, total_descriptor, total_matches);

	//Refine the matches
	std::vector<std::vector<cv::DMatch>> total_RefinedMatches; 
	RefineMatches(total_matches, total_RefinedMatches, ratio);

	
	cv::Mat matched_images;

	for (int i = 0; i < images.size() - 1; i++) {

		//Draw the matches of keypoints between two images
		drawMatches(images[i], total_keypoints[i], images[i + 1], total_keypoints[i + 1], total_RefinedMatches[i], matched_images);
		imshow("Matched Images: ", matched_images);
		cv::waitKey(0);

	}

	//Store good matches
	std::vector<std::vector<cv::DMatch>> total_goodMatches; 
	findGoodMatches(total_keypoints, total_RefinedMatches, total_goodMatches);

	//Find the pixel distance
	std::vector<std::vector<float>> totalDistance;
	findDistance(images, totalDistance, total_keypoints, total_goodMatches);

	std::vector<float> total_MeanDistance;
	cv::Mat panoramic;
	//Merge images 
	merge(images, panoramic, total_MeanDistance, totalDistance, total_goodMatches);

	cv::imwrite("C:/Users/yukse/OneDrive/Masaüstü/Lectures/Computer Vision/LAB 4/panoramic.png", panoramic);
	
	return 0;
}
