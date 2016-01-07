#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include "FeatureDetectorSIFT.h"
#include "Image.h"
#include "KeypointDescriptor.h"

using namespace SceneComps;

int main( int argc, char** argv ) {

	if (argc != 2) { 
		std::cout << "Must provide filepath argument.\n";
		return -1; 
	}

	// Create image
	std::string filepath = argv[1];
	Image image = Image(filepath);
    
    // Detect keypoints
    FeatureDetectorSIFT siftDetector = FeatureDetectorSIFT();
    std::vector<KeypointDescriptor> sift_keypoints = siftDetector.detect(image);

    // Convert descriptors back to cv keypoints because that's what drawKeypoints takes :(
    std::vector<KeyPoint> keypoints(sift_keypoints.begin(), sift_keypoints.end());

    // Draw keypoints
    Mat detectedImage;
    drawKeypoints(image.matrix, keypoints, detectedImage, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

    // Show result
	namedWindow( "SIFT", 0 );
	imshow("SIFT", detectedImage(Rect(0, 0, image.width(), image.height() )));

	std::cout << "^C to exit.\n";
	for(;;){
        waitKey(0);
    }

	return 0;
}