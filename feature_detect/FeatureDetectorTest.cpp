#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include "FeatureDetector.h"
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

	cv::Size s = image.matrix.size();
	int height = s.height;
	int width = s.width;

	namedWindow( "SIFT", 0 );
	imshow("SIFT", image.matrix( Rect(0,0,width,height) ));

	// Don't close immediately
	for(;;){
		waitKey(0);
	}

	return 0;
}