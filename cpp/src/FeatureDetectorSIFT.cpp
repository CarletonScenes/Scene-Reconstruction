#include <iostream>
#include "FeatureDetectorSIFT.h"
#include "OpenCV.h"
#include "KeypointDescriptor.h"

using namespace cv;
using namespace cv::xfeatures2d;

namespace SceneComps {

std::vector<KeypointDescriptor> FeatureDetectorSIFT::detect(Image image) {

    Mat imgMatrix = image.matrix;

    // Use OpenCV SIFT detector
    std::vector<KeyPoint> keyPoints;
    Ptr<SIFT> sift_detector = SIFT::create(sift_minHessian);
    sift_detector->detect(imgMatrix, keyPoints);

    std::vector<KeypointDescriptor> descriptors;

    // Convert to Keypoint Descriptors (for no reason at this point...)
    for (std::vector<KeyPoint>::iterator it = keyPoints.begin(); it != keyPoints.end(); ++it) {
    	descriptors.push_back(KeypointDescriptor(*it));
    }

    return descriptors;
};

}
