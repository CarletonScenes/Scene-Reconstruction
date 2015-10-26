#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <cmath>
#include "FeatureDetectorSIFT.h"
#include "KeyPointDescriptor.h"
#include "Image.h"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"


using namespace cv;
using namespace cv::xfeatures2d;
using namespace SceneComps;
using namespace std;

vector<DMatch> matches;

vector<DMatch> filteredMatches;
vector<DMatch> emptyMatches;

Mat filtered_matches_matrix;

vector<KeyPoint> sift_keypoints1;
vector<KeyPoint> sift_keypoints2;

Image image1("");
Image image2("");
int matchColor = 250;
int pointColor = 0;

void onMouse(int event, int x, int y, int flags, void* userdata) {
    if ( event == EVENT_MOUSEMOVE ) {

        // Index and distance of closest keypoint
        int closest = 0;
        float minDistance = 1000;

        // Find closest keypoint
        int index = 0;
        for (vector<KeyPoint>::iterator it = sift_keypoints1.begin(); it != sift_keypoints1.end(); ++it) {
            int kpx = it->pt.x;
            int kpy = it->pt.y;
            float distance = sqrt(pow(x-kpx, 2) + pow(y-kpy, 2));
            // cout << "iterating\n";
            if (distance < minDistance) {
                minDistance = distance;
                closest = index;
            }
            index ++;
        }

        vector<DMatch> line;

        // If in distance threshold, find the matching. Otherwise, draw an empty matching
        if (minDistance < 20) {
            for (vector<DMatch>::iterator it = filteredMatches.begin(); it != filteredMatches.end(); ++it) {
                if (it->queryIdx == closest) {
                    line.push_back(*it);
                }
            }
        }
        drawMatches(image1.matrix, sift_keypoints1, image2.matrix, sift_keypoints2, line, filtered_matches_matrix, matchColor, pointColor);
        imshow("filtered_matches", filtered_matches_matrix);
    }
}

int main( int argc, char** argv ) {
    
	if (argc != 3) { 
		cout << "Must provide filepath argument.\n";
		return -1; 
	}

	// Create image
    string filepath1 = argv[1];
    image1 = Image(filepath1);
    string filepath2 = argv[2];
    image2 = Image(filepath2);
    
    // Detect keypoints
    FeatureDetectorSIFT siftDetector = FeatureDetectorSIFT();
    vector<KeyPoint> keypoints1 = siftDetector.detect(image1);
    vector<KeyPoint> keypoints2 = siftDetector.detect(image2);

    sift_keypoints1 = vector<KeyPoint>(keypoints1.begin(), keypoints1.end());
    sift_keypoints2 = vector<KeyPoint>(keypoints2.begin(), keypoints2.end());

    //STUFF FROM THE OPEN CV EXAMPLE BELOW
    // https://github.com/npinto/opencv/blob/master/samples/cpp/matcher_simple.cpp
    cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
    
    Mat descriptors1, descriptors2; 
    f2d->compute(image1.matrix, sift_keypoints1, descriptors1);
    f2d->compute(image2.matrix, sift_keypoints2, descriptors2);
    
    BFMatcher matcher;
    matcher.match(descriptors1, descriptors2, matches);
    
    vector<Point2f> ptList1;
    vector<Point2f> ptList2;
    
    vector<int> queryIdxs;
    vector<int> trainIdxs;
    
    for (vector<DMatch>::size_type i = 0; i != matches.size(); i++){
        queryIdxs.push_back(matches[i].queryIdx);
        trainIdxs.push_back(matches[i].trainIdx);
    }
    
    KeyPoint::convert(sift_keypoints1, ptList1, queryIdxs);
    KeyPoint::convert(sift_keypoints2, ptList2, trainIdxs);
    
    vector<uchar> funOut;
    
    //press 8 for RANSAC
    Mat F = findFundamentalMat(ptList1, ptList2, 8, 3, .99, funOut);
    
    vector<int> funOutInt(funOut.begin(), funOut.end());
    
    for (vector<int>::size_type i = 0; i != funOut.size(); i++){
        cout << funOutInt[i];
        if (funOutInt[i]==1){
            filteredMatches.push_back(matches[i]);
        }
    }
    
    
    // drawing the results
    // namedWindow("matches", 1);
    
    // drawMatches(image1.matrix, sift_keypoints1, image2.matrix, sift_keypoints2, matches, img_matches);
    // drawMatches(image1.matrix, sift_keypoints1, image2.matrix, sift_keypoints2, filteredMatches, filtered_matches_matrix);
    // imshow("matches", img_matches);
    
    namedWindow("filtered_matches", 1);
    setMouseCallback("filtered_matches", onMouse, NULL);
    drawMatches(image1.matrix, sift_keypoints1, image2.matrix, sift_keypoints2, emptyMatches, filtered_matches_matrix, matchColor, pointColor);
    imshow("filtered_matches", filtered_matches_matrix);

    cout << "^C to exit.\n";
    waitKey(0);

	return 0;
}
