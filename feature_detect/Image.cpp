#include "Image.h"
#include "OpenCV.h"
#include <stdio.h>
#include <unistd.h>
#include <iostream>

using namespace cv;

namespace SceneComps {

Image::Image(std::string path) {
    fileLocation = path;
    matrix = imread( fileLocation, CV_LOAD_IMAGE_COLOR);
}

std::vector<int> Image::getExif(void){ 
    return exif;
}

void Image::setFlags(std::vector<int> flags) {
    flags = flags;
}

}