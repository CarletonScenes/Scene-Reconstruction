#pragma once
#include "OpenCV.h"

using namespace cv;

namespace SceneComps {

class Image {
    
    public:
        void setFlags(std::vector<int>);
        Mat getMatrix();
        std::vector<int> getExif();
        Image(std::string fileLocation); 
        Mat matrix;

        // Get height and width?

    private:
        std::vector<int> flags;
        std::string fileLocation;
 
        // OpenCV doesn't support exif. we'll figure that out
        std::vector<int> exif;
};

}