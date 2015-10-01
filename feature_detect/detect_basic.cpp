// Edited from: http://docs.opencv.org/3.0-last-rst/doc/tutorials/features2d/feature_detection/feature_detection.html#feature-detection

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

  // SIFT
  //-- Step 1: Detect the keypoints using SIFT Detector
  int sift_minHessian = 10000;
  Ptr<SIFT> sift_detector = SIFT::create( sift_minHessian );
  std::vector<KeyPoint> sift_keypoints;
  sift_detector->detect( img_1, sift_keypoints );

  //-- Draw keypoints
  Mat img_sift_keypoints;
  drawKeypoints( img_1, sift_keypoints, img_sift_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

  //-- Show detected (drawn) keypoints
  imshow("SIFT", img_sift_keypoints );
  moveWindow("SIFT", 0, 0);

  // SURF
  //-- Step 1: Detect the keypoints using SURF Detector
  int surf_minHessian = 400;
  Ptr<SURF> surf_detector = SURF::create( surf_minHessian );
  std::vector<KeyPoint> surf_keypoints;
  surf_detector->detect( img_1, surf_keypoints );

  //-- Draw keypoints
  Mat img_surf_keypoints;
  drawKeypoints( img_1, surf_keypoints, img_surf_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

  //-- Show detected (drawn) keypoints
  imshow("SURF", img_surf_keypoints );
  moveWindow("SURF", img_1.cols,0);

  // ORB
  //-- Step 1: Detect the keypoints using ORB Detector
  int orb_minHessian = 10000;
  Ptr<ORB> orb_detector = ORB::create( orb_minHessian );
  std::vector<KeyPoint> orb_keypoints;
  orb_detector->detect( img_1, orb_keypoints );

  //-- Draw keypoints
  Mat img_orb_keypoints;
  drawKeypoints( img_1, orb_keypoints, img_orb_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

  //-- Show detected (drawn) keypoints
  imshow("ORB", img_orb_keypoints );
  moveWindow("ORB", img_1.cols*2, 0);

  // // FAST
  // //-- Step 1: Detect the keypoints using FAST Detector
  // int minHessian = 10000;
  // Ptr<FAST> detector = FAST::create( minHessian );
  // std::vector<KeyPoint> fast_keypoints;
  // detector->detect( img_1, fast_keypoints );

  // //-- Draw keypoints
  // Mat img_fast_keypoints;
  // drawKeypoints( img_1, fast_keypoints, img_fast_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

  // //-- Show detected (drawn) keypoints
  // imshow("FAST", img_fast_keypoints );



  waitKey(0);

  return 0;
}
