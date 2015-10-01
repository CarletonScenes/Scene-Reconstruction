#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace cv::xfeatures2d;

int main( int argc, char** argv )
{
  if( argc != 2 )
  { return -1; }

  Mat img_1 = imread( argv[1], IMREAD_GRAYSCALE );

  if( !img_1.data )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 400;

  Ptr<SURF> detector = SURF::create( minHessian );

  std::vector<KeyPoint> keypoints_1;

  detector->detect( img_1, keypoints_1 );

  //-- Draw keypoints
  Mat img_keypoints_1;

  drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

  //-- Show detected (drawn) keypoints
  imshow("Keypoints 1", img_keypoints_1 );

  waitKey(0);

  return 0;
}
