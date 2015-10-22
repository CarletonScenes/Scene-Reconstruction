#pragma once
#include <stdio.h>
#include <unistd.h>
#include "Image.h"
#include "OpenCV.h"


namespace SceneComps{
    class KeypointDescriptor: public cv::KeyPoint {
        public:
            KeypointDescriptor() : cv::KeyPoint () {};
            Ptr<Image> image;       
    };

}