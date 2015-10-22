#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include "FeatureDetectorSIFT.h"
#include "Image.h"


using namespace cv;
using namespace cv::xfeatures2d;
using namespace SceneComps;

//int main( int argc, char** argv ) {
//
//	if (argc != 2) { 
//		std::cout << "Must provide filepath argument.\n";
//		return -1; 
//	}
//
//	// Create image
//	std::string filepath1 = argv[1];
//	Image image1 = Image(filepath1);
//    std::string filepath2 = argv[2];
//	Image image2 = Image(filepath2);
//    
//    // Detect keypoints
//    FeatureDetectorSIFT siftDetector = FeatureDetectorSIFT();
//    std::vector<KeyPoint> sift_keypoints1 = siftDetector.detect(image1);
//    std::vector<KeyPoint> sift_keypoints2 = siftDetector.detect(image2);
// 
//    // Draw keypoints
//    Mat detectedImage1;
//    Mat detectedImage2;
//    drawKeypoints(image1.matrix, sift_keypoints1, detectedImage1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
//    
//    // Show result
//	namedWindow( "SIFT", 0 );
//	imshow("SIFT", detectedImage1(Rect(0, 0, image1.width(), image1.height() )));
//
//	std::cout << "Press any key to exit.\n";
//	  for(;;){
//    waitKey(0);
//
//  }
//
//	return 0;
//}
