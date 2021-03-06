#pragma once
#include "FeatureDetector.h"

namespace SceneComps {

class FeatureDetectorSIFT : public FeatureDetector {
    public:  
        FeatureDetectorSIFT () : FeatureDetector () {};
        std::vector<KeypointDescriptor> detect(Image image);

    private:
        const static int sift_minHessian = 10000;
};

}
