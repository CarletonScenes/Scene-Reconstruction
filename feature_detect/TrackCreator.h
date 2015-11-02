#pragma once
#include <stdio.h>
#include <unistd.h>
#include "Image.h"
#include "OpenCV.h"
#include <cmath>
#include "FeatureDetectorSIFT.h"
#include "KeypointDescriptor.h"
#include "Image.h"
#include "opencv2/calib3d.hpp"
#include "KeypointMetadata.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

namespace SceneComps{
    
class TrackCreator {
    public:
        TrackCreator() {};
        static void computeTracks(vector<Image> images, vector<KeypointMetadata> pointList);
        static void computeAndFilterPairMatches(vector<KeyPoint> &key1, vector<KeyPoint> &key2, Image
                                                image1, Image image2, vector<DMatch> &matches); 
        static void findAndFilterSubgraphs(vector<KeypointMetadata> &pointList);
    private:
        static void traverseFromPoint(vector<KeypointMetadata> &pointList, vector<int> &currentTrackIndices, vector<KeypointMetadata>::size_type index);
};
    
   
}