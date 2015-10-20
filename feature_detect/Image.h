#pragma once
#include "OpenCV.h"

using namespace cv;

namespace SceneComps {

class Image {
    
    public:
        void setFlags(std::vector<int>);
        std::vector<int> getExif();
        Image(std::string fileLocation); 
    private:
        std::vector<int> flags;
        std::string fileLocation;
        Mat matrix;
 
        // OpenCV doesn't support exif. we'll figure that out
         std::vector<int> exif;
};

}