#include "FeatureDetectorSIFT.h"

namespace SceneComps{

std::vector<KeyPoint> FeatureDetectorSIFT::detect(Image image)
        {
            Mat img_1 = image.matrix;
            std::vector<KeyPoint> keyPoints;
            Ptr<SIFT> sift_detector = SIFT::create( sift_minHessian );
            sift_detector->detect( img_1, keyPoints );
            return keyPoints;
        };
        
}