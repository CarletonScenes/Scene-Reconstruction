#include "FeatureDetector.h"
#include "Image.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

namespace SceneComps {

class FeatureDetectorSIFT : public FeatureDetector {
    public:  
        FeatureDetectorSIFT () : FeatureDetector () {};
        
        //Detects features and stores them by modifying vector
        void detect(Image image, std::vector<KeyPoint> keyPoints)
        {
            Mat img_1 = image.matrix;
            Ptr<SIFT> sift_detector = SIFT::create( sift_minHessian );
            sift_detector->detect( img_1, keyPoints );
        };
    private:
        int sift_minHessian = 10000;
};

}