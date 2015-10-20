using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

class FeatureDetector {

public:
    virtual void Detect(Image image, std::vector<KeyPoint>) = 0;
};