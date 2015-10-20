using namespace std;
using namespace cv;

class Image{
    
    public:
        void setFlags( std::vector<int>);
        std::vector<int> getExif();
        Line(string fileLocation); 
    private:
        std::vector<int> flags;
        string fileLocation;
        Mat matrix;
 
        // OpenCV doesn't support exif. we'll figure that out
         std::vector<int> exif;
};

Image::Image(string file){
    fileLocation = file;
    matrix = imread( fileLocation, CV_LOAD_IMAGE_COLOR);
}

std::vector<int> Image::getExif(void){
    return exif;
}

void Image::setFlags(std::vector<int> flags){
    flags = flags;
}