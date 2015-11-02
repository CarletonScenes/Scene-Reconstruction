#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <cmath>
#include "FeatureDetectorSIFT.h"
#include "KeypointDescriptor.h"
#include "Image.h"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "KeypointMetadata.h"
#include "TrackCreator.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

namespace SceneComps {
    
void TrackCreator::computeTracks(vector<Image> images, vector<KeypointMetadata> pointList) {
    
    vector<KeyPoint> key1;
    vector<KeyPoint> key2;
    Image image1 = images[0];
    Image image2 = images[1]; 
    vector<DMatch> matches;
    KeypointMetadata point1;
    KeypointMetadata point2;
    int totalmatches = 0;
    int repeats;
    int different;
    
    //For every pair of images, match images
    for (vector<Image>::size_type i = 0; i != images.size(); i++){
        for (vector<Image>::size_type j = i+1; j != images.size(); j++){
            key1.clear();
            key2.clear();
            image1 = images[i];
            image2 = images[j];
            matches.clear();
            
            TrackCreator::computeAndFilterPairMatches(key1, key2, image1, image2, matches);
            
//            matches.clear();
//            matches.push_back(DMatch(0,0,0));
//            matches.push_back(DMatch(1,1,0));
//            matches.push_back(DMatch(2,2,0));
            
            //Add all keypoints to pointlist, if this is our first time seeing them
            if (i==0){
                if (j==1){   
                    for (vector<KeyPoint>::size_type a = 0; a != key1.size(); a++){
                        pointList.push_back(KeypointMetadata(a, i));
                    }
                }
                for (vector<KeyPoint>::size_type b = 0; b != key2.size(); b++){
                    pointList.push_back(KeypointMetadata(b, j));
                }
            }
            
            //for each entry in matches between images i and j
            for (vector<DMatch>::size_type k = 0; k != matches.size(); k++){
                totalmatches ++;
                //look at all the points
                for (vector<KeypointMetadata>::size_type p = 0; p != pointList.size(); p++){
                    // if it has the same image and the same point2d, 
                    //then it's the container for the match's first point
                    if (pointList[p].imageIndex == i &&  pointList[p].pointIndex == matches[k].queryIdx ){
                         //find the container matching the point matched in image j
                        for (vector<KeypointMetadata>::size_type q = 0; q != pointList.size(); q++){  
                            if (pointList[q].imageIndex == j &&  pointList[q].pointIndex == matches[k].trainIdx){
                                pointList[q].matchIndexList.push_back(p);
                                pointList[p].matchIndexList.push_back(q);
                                //cout << "matching edge " << p << " with " << q <<"\n";
                            }
                        }
                    }
                }
            }
            
        }
    }
    findAndFilterSubgraphs(pointList);
    
    
    //print results
    int matched = 0;
    int unmatched = 1;
    for (vector<KeypointMetadata>::size_type p = 0; p != pointList.size(); p++){
        cout << pointList[p].imageIndex << ":" << pointList[p].trackNumber << " ";
//        cout << pointList[p].imageIndex << ":";
//        if (pointList[p].matchIndexList.size() > 0){
//            for (vector<int>::size_type m = 0; m != pointList[p].matchIndexList.size(); m++){
//                cout << pointList[pointList[p].matchIndexList[m]].imageIndex;
//            }
//            cout << " ";
//            matched ++;
//        } else {
//            cout << "_ ";
//            unmatched ++;
//        }
    }
    cout << "\n" << pointList.size() << "\n" <<matched << "matched, " << unmatched << "unmatched \n"; 
    cout << "total matches: " << totalmatches ;
    
} 
    
void TrackCreator::findAndFilterSubgraphs(vector<KeypointMetadata> &pointList){
    //Takes the pointList and figures out the trackNumber and matched of each point
    int currentTrack = 1;
    vector<int> currentTrackIndices;
    vector<int> imagesUsed;
    int repeatedImage = 0;
    //go through pointList
    for (vector<KeypointMetadata>::size_type i = 0; i != pointList.size(); i++){
        currentTrackIndices.clear();
        imagesUsed.clear();
        if (pointList[i].visited==0){
            traverseFromPoint(pointList, currentTrackIndices, i);
            if (currentTrackIndices.size() > 1){
                repeatedImage = 0;
                for (vector<int>::size_type j = 0; j != currentTrackIndices.size(); j++){
                    //if this point's image hasn't been seen in this track
                    if (find(imagesUsed.begin(),imagesUsed.end(),pointList[currentTrackIndices[j]].imageIndex) == imagesUsed.end()){
                        //add the image to list of used images
                        imagesUsed.push_back(pointList[currentTrackIndices[j]].imageIndex);
                    } else {
                        repeatedImage = 1;
                    }
                }
                //none of the currentTrack points used the same image twice
                if (repeatedImage == 0){
                    for (vector<int>::size_type k = 0; k != currentTrackIndices.size(); k++){
                        pointList[currentTrackIndices[k]].trackNumber = currentTrack;
                    }
                    currentTrack ++;
                //there's a repeat image
                } else {
                    for (vector<int>::size_type k = 0; k != currentTrackIndices.size(); k++){
                        pointList[currentTrackIndices[k]].trackNumber = -1;
                    }
                }
            }
        }
    }
}
    
void TrackCreator::traverseFromPoint(vector<KeypointMetadata> &pointList, vector<int> &currentTrackIndices, vector<KeypointMetadata>::size_type index){
    currentTrackIndices.push_back(index);
    pointList[index].visited = 1;
    for (vector<int>::size_type i = 0; i != pointList[index].matchIndexList.size(); i++){
        if (pointList[pointList[index].matchIndexList[i]].visited == 0){
            traverseFromPoint(pointList, currentTrackIndices, pointList[index].matchIndexList[i]);
        }
    }
}
    
void TrackCreator::computeAndFilterPairMatches(vector<KeyPoint> &key1, vector<KeyPoint> &key2, Image image1, Image image2, vector<DMatch> &matches){
    
    FeatureDetectorSIFT siftDetector = FeatureDetectorSIFT();
    vector<KeypointDescriptor> keypoints1 = siftDetector.detect(image1);
    vector<KeypointDescriptor> keypoints2 = siftDetector.detect(image2);

    // Convert descriptors back to cv keypoints :(
    key1.assign(keypoints1.begin(), keypoints1.end());
    key2.assign(keypoints2.begin(), keypoints2.end());
    //vector<KeyPoint> sift_keypoints2 = vector<KeyPoint>(keypoints2.begin(), keypoints2.end());

    //STUFF FROM THE OPEN CV EXAMPLE BELOW
    // https://github.com/npinto/opencv/blob/master/samples/cpp/matcher_simple.cpp
    cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
    
    Mat descriptors1, descriptors2; 
    f2d->compute(image1.matrix, key1, descriptors1);
    f2d->compute(image2.matrix, key2, descriptors2);
    
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
    
    KeyPoint::convert(key1, ptList1, queryIdxs);
    KeyPoint::convert(key2, ptList2, trainIdxs);
    
    vector<uchar> funOut;
    
    //press 8 for RANSAC
    Mat F = findFundamentalMat(ptList1, ptList2, 8, 3, .99, funOut);
    
    vector<int> funOutInt(funOut.begin(), funOut.end());
    vector<DMatch> filtered_matches;
    
    for (vector<int>::size_type i = 0; i != funOut.size() ; i++){
        if (funOutInt[i]==0){
            filtered_matches.push_back(matches[i]);
        }
    }
    
    int repeatsTrain = 0;
    int repeatsQuery = 0;
    int different = 0;
    
     //for testing repeats
    for (vector<DMatch>::size_type x = 0; x != filtered_matches.size(); x++){
        for (vector<DMatch>::size_type y = x+1; y != filtered_matches.size(); y++){
            if (filtered_matches[x].trainIdx == filtered_matches[y].trainIdx){
                repeatsTrain++;
            }
            else if (filtered_matches[x].queryIdx == filtered_matches[y].queryIdx){
                repeatsQuery++;
            }
            else {
                different++;
            }
        }
    }
            cout << "Qrepeats: " << repeatsQuery <<" | Trepeats: " << repeatsTrain << " | different: "<< different << "\n";
    
    matches.clear();
    matches.assign(filtered_matches.begin(),filtered_matches.end());
    
}    
    
    
}