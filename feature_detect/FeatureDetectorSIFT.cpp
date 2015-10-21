#include "FeatureDetectorSIFT.h"
#include "OpenCV.h"

using namespace cv;
using namespace cv::xfeatures2d;

namespace SceneComps {

std::vector<KeyPoint> FeatureDetectorSIFT::detect(Image image) {

    Mat imgMatrix = image.matrix;
    std::vector<KeyPoint> keyPoints;

    // Use OpenCV SIFT detector
    Ptr<SIFT> sift_detector = SIFT::create(sift_minHessian);
    sift_detector->detect(imgMatrix, keyPoints);

    return keyPoints;
};

}