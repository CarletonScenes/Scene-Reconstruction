#pragma once
#include "OpenCV.h"
#include "Image.h"

namespace SceneComps {

class FeatureDetector {

	public:
	    virtual std::vector<KeyPoint> detect(Image image) = 0;
	    
};

}