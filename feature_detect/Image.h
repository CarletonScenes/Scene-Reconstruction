#pragma once
#include "OpenCV.h"

using namespace cv;

namespace SceneComps {

class Image {
    
    public:
        Image(std::string fileLocation); 
        Image() {};
        int height();
        int width();

        Mat matrix;

        // OpenCV doesn't support exif. we'll figure that out
        std::vector<int> exif;
        std::vector<int> flags;
        std::string fileLocation;
              
};

}