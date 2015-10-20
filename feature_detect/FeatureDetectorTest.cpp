#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include "FeatureDetectorSIFT.h"
#include "Image.h"


// int WINDOW_WIDTH = 640;
// int WINDOW_HEIGHT = 640;

using namespace SceneComps;

int main( int argc, char** argv ) {

	// Make sure image path is included
	if( argc != 2 )
	{ return -1; }

	std::string filepath = argv[1];

	// Right now, just opens and displays an image
	Image image = Image(filepath);
    
    FeatureDetectorSIFT siftDetector = FeatureDetectorSIFT();
    
    std::vector<KeyPoint> sift_keypoints = siftDetector.detect(image);
    //-- Draw keypoints
    Mat detectedImage;
    drawKeypoints( image.matrix, sift_keypoints, detectedImage, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    
    int length = sift_keypoints.size();

	cv::Size s = detectedImage.size();
	int height = s.height;
	int width = s.width;

	namedWindow( "SIFT", 0 );
	imshow("SIFT", detectedImage( Rect(0,0,width,height) ));

	// Don't close immediately
	for(;;){
		waitKey(0);
	}

	return 0;
}