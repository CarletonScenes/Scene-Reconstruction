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

int Image::height() {
    return matrix.size().height;
}

int Image::width() {
    return matrix.size().width;
}

}