#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <cmath>
#include <map>
#include <set>

#include "FeatureDetectorSIFT.h"
#include "KeyPointDescriptor.h"
#include "Image.h"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

#include "exif.h"

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

void GetFilesInDirectory(vector<string> &out, const string &directory)
{
    DIR *dir;
    class dirent *ent;
    class stat st;

    dir = opendir(directory);
    while ((ent = readdir(dir)) != NULL) {
        const string file_name = ent->d_name;
        const string full_file_name = directory + "/" + file_name;

        if (file_name[0] == '.')
            continue;

        if (stat(full_file_name.c_str(), &st) == -1)
            continue;

        const bool is_directory = (st.st_mode & S_IFDIR) != 0;

        if (is_directory)
            continue;

        out.push_back(full_file_name);
    }
    closedir(dir);
}

int main( int argc, char** argv ) {
    
  if (argc != 2) { 
    cout << "Must provide directory argument.\n";
    return -1; 
  }


  vector<string> files;
  GetFilesInDirectory(files, argv[1]);

  int originalIndex = 0;
  int imgAindex = 0;
  int imgBindex = 0;

  std::set<int> indexesIncluded;
  std::map<int, vector<Mat,Mat>> knownRts;

  // Find first two images based on snavely method - set originalIndex, imgAindex, imgBindex

  indexesIncluded.insert(imgAindex);
  indexesIncluded.insert(imgBindex);

  while (indexesIncluded.size() != files.size()) {
      // find features in each image, find matches

      // findEssentialMatrix

      // recoverPose between A and B

      // convert R|t for B using original R|t value for A if we have it. (check knownRts map)

      // add new R|ts to the map for both images

      // triangulatePoints and add to cloud

      // find next B to use based on best match between remaining images (Snavely's method) and an included image.
  }



  // Create image
    string filepath1 = argv[1];
    image1 = Image(filepath1);
    string filepath2 = argv[2];
    image2 = Image(filepath2);
    
    // Detect keypoints
    FeatureDetectorSIFT siftDetector = FeatureDetectorSIFT();
    vector<KeypointDescriptor> keypoints1 = siftDetector.detect(image1);
    vector<KeypointDescriptor> keypoints2 = siftDetector.detect(image2);

    // Convert descriptors back to cv keypoints :(
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
        if (funOutInt[i]==1){
            filteredMatches.push_back(matches[i]);
        }
    }
    
    namedWindow("filtered_matches", 1);
    drawMatches(image1.matrix, sift_keypoints1, image2.matrix, sift_keypoints2, emptyMatches, filtered_matches_matrix, matchColor, pointColor);
    imshow("filtered_matches", filtered_matches_matrix);

    cout << "^C to exit.\n";
    waitKey(0);

  return 0;
}
