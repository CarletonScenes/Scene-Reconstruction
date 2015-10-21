// Edited from: http://docs.opencv.org/3.0-last-rst/doc/tutorials/features2d/feature_detection/feature_detection.html#feature-detection

#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

using namespace std;

using namespace cv;
using namespace cv::xfeatures2d;

int WINDOW_WIDTH = 640;
int WINDOW_HEIGHT = 640;

int MOUSE_ADJUST_FACTOR = (float) 1/120;

Mat img_sift_keypoints;
Mat img_surf_keypoints;
Mat img_orb_keypoints;

static void onMouse( int event, int x, int y, int, void* )
{
    cout << "mouse event";
    if( event != EVENT_MOUSEWHEEL || event != EVENT_MOUSEHWHEEL )
        return;

    int heightDelta = getMouseWheelDelta(EVENT_MOUSEWHEEL) * MOUSE_ADJUST_FACTOR;
    int widthDelta  = getMouseWheelDelta(EVENT_MOUSEHWHEEL) * MOUSE_ADJUST_FACTOR;

    imshow("SIFT", img_sift_keypoints( Rect(widthDelta, heightDelta, WINDOW_WIDTH,WINDOW_HEIGHT) ));
    imshow("SURF", img_surf_keypoints( Rect(widthDelta, heightDelta, WINDOW_WIDTH,WINDOW_HEIGHT) ));
    imshow("ORB",  img_orb_keypoints(  Rect(widthDelta, heightDelta, WINDOW_WIDTH,WINDOW_HEIGHT) ));

    cout << "scroll detected: " << widthDelta << " " << heightDelta;

    // Point seed = Point(x,y);
    // int lo = ffillMode == 0 ? 0 : loDiff;
    // int up = ffillMode == 0 ? 0 : upDiff;
    // int flags = connectivity + (newMaskVal << 8) +
    //             (ffillMode == 1 ? FLOODFILL_FIXED_RANGE : 0);
    // int b = (unsigned)theRNG() & 255;
    // int g = (unsigned)theRNG() & 255;
    // int r = (unsigned)theRNG() & 255;
    // Rect ccomp;

    // Scalar newVal = isColor ? Scalar(b, g, r) : Scalar(r*0.299 + g*0.587 + b*0.114);
    // Mat dst = isColor ? image : gray;
    // int area;

    // if( useMask )
    // {
    //     threshold(mask, mask, 1, 128, THRESH_BINARY);
    //     area = floodFill(dst, mask, seed, newVal, &ccomp, Scalar(lo, lo, lo),
    //               Scalar(up, up, up), flags);
    //     imshow( "mask", mask );
    // }
    // else
    // {
    //     area = floodFill(dst, seed, newVal, &ccomp, Scalar(lo, lo, lo),
    //               Scalar(up, up, up), flags);
    // }

    // imshow("image", dst);
    // cout << area << " pixels were repainted\n";
}

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
  drawKeypoints( img_1, sift_keypoints, img_sift_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

  // SURF
  //-- Step 1: Detect the keypoints using SURF Detector
  int surf_minHessian = 400;
  Ptr<SURF> surf_detector = SURF::create( surf_minHessian );
  std::vector<KeyPoint> surf_keypoints;
  surf_detector->detect( img_1, surf_keypoints );

  //-- Draw keypoints
  drawKeypoints( img_1, surf_keypoints, img_surf_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

  // ORB
  //-- Step 1: Detect the keypoints using ORB Detector
  int orb_minHessian = 10000;
  Ptr<ORB> orb_detector = ORB::create( orb_minHessian );
  std::vector<KeyPoint> orb_keypoints;
  orb_detector->detect( img_1, orb_keypoints );

  //-- Draw keypoints
  drawKeypoints( img_1, orb_keypoints, img_orb_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

  // // FAST
  // //-- Step 1: Detect the keypoints using FAST Detector
  // int minHessian = 10000;
  // Ptr<FAST> detector = FAST::create( minHessian );
  // std::vector<KeyPoint> fast_keypoints;
  // detector->detect( img_1, fast_keypoints );

  // //-- Draw keypoints
  // Mat img_fast_keypoints;
  // drawKeypoints( img_1, fast_keypoints, img_fast_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

  //-- Show detected (drawn) keypoints
  namedWindow( "SIFT", 0 );
  namedWindow( "SURF", 0 );
  namedWindow( "ORB", 0 );

  imshow("SIFT", img_sift_keypoints( Rect(0,0,WINDOW_WIDTH,WINDOW_HEIGHT) ));
  moveWindow("SIFT", 0, 0);
  imshow("SURF", img_surf_keypoints( Rect(0,0,WINDOW_WIDTH,WINDOW_HEIGHT) ) );
  moveWindow("SURF", img_1.cols,  0);
  imshow("ORB", img_orb_keypoints( Rect(0,0,WINDOW_WIDTH,WINDOW_HEIGHT) ));
  moveWindow("ORB", img_1.cols * 2, 0);

  // createTrackbar("Hscroll", "SIFT", 0, 15, scrollWindow);
  setMouseCallback( "SIFT", onMouse, 0 );
  setMouseCallback( "SURF", onMouse, 0 );
  setMouseCallback( "ORB",  onMouse, 0 );

  // //-- Show detected (drawn) keypoints
  // imshow("FAST", img_fast_keypoints );

  for(;;){
    waitKey(0);

  }
  

  // while(1){
  //   usleep(1000);
  // }

  return 0;
}
