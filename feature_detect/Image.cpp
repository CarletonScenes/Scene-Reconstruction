#import "Image.h";
#include "OpenCV.h"

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