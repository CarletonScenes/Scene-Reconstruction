#pragma once
#include "OpenCV.h"
#include "Image.h"
#include "KeypointDescriptor.h"

namespace SceneComps {

class FeatureDetector {

	public:
	    virtual std::vector<KeypointDescriptor> detect(Image image) = 0;
	    
};

}