#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include "FeatureDetectorSIFT.h"
#include "Image.h"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"


using namespace cv;
using namespace cv::xfeatures2d;
using namespace SceneComps;

int main( int argc, char** argv ) {
    
	if (argc != 3) { 
		std::cout << "Must provide filepath argument.\n";
		return -1; 
	}

	// Create image
	std::string filepath1 = argv[1];
	Image image1 = Image(filepath1);
    std::string filepath2 = argv[2];
	Image image2 = Image(filepath2);
    
    // Detect keypoints
    FeatureDetectorSIFT siftDetector = FeatureDetectorSIFT();
    std::vector<KeyPoint> sift_keypoints1 = siftDetector.detect(image1);
    std::vector<KeyPoint> sift_keypoints2 = siftDetector.detect(image2);

    //STUFF FROM THE OPEN CV EXAMPLE BELOW
    // https://github.com/npinto/opencv/blob/master/samples/cpp/matcher_simple.cpp
    cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
    
    Mat descriptors1, descriptors2; 
    f2d->compute(image1.matrix, sift_keypoints1, descriptors1);
    f2d->compute(image2.matrix, sift_keypoints2, descriptors2);
    
    BFMatcher matcher;
    std::vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    
    std::vector<Point2f> ptList1;
    std::vector<Point2f> ptList2;
    
    std::vector<int> queryIdxs;
    std::vector<int> trainIdxs;
    
    for (std::vector<DMatch>::size_type i = 0; i != matches.size(); i++){
        queryIdxs.push_back(matches[i].queryIdx);
        trainIdxs.push_back(matches[i].trainIdx);
    }
    
    KeyPoint::convert(sift_keypoints1, ptList1, queryIdxs);
    KeyPoint::convert(sift_keypoints2, ptList2, trainIdxs);
    
    std::vector<uchar> funOut;
    
    //press 8 for RANSAC
    Mat F = findFundamentalMat(ptList1, ptList2, 8, 3, .99, funOut);
    
    std::vector<int> funOutInt(funOut.begin(), funOut.end());
    std::vector<DMatch> filteredMatches;
    
    for (std::vector<int>::size_type i = 0; i != funOut.size(); i++){
        std::cout << funOutInt[i];
        if (funOutInt[i]==1){
            filteredMatches.push_back(matches[i]);
        }
    }
    
    
    // drawing the results
    namedWindow("matches", 1);
    namedWindow("filteredMatches", 1);
    Mat img_matches;
    Mat filtered_matches;
    drawMatches(image1.matrix, sift_keypoints1, image2.matrix, sift_keypoints2, matches, img_matches);
    drawMatches(image1.matrix, sift_keypoints1, image2.matrix, sift_keypoints2, filteredMatches, filtered_matches);
    imshow("matches", img_matches);
    imshow("Filtered matches", filtered_matches);

	std::cout << "Press any key to exit.\n";
	  for(;;){
    waitKey(0);

  }

	return 0;
}
