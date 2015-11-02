#pragma once
#include <stdio.h>
#include <unistd.h>
#include "Image.h"
#include "OpenCV.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

namespace SceneComps {
   class KeypointMetadata {
       public:
            KeypointMetadata() {};
			KeypointMetadata(int kp, int im) {
				pointIndex = kp;
                imageIndex = im;
                visited = 0;
                trackNumber = 0;
                matchIndexList = vector<int>();
        	};
            int pointIndex;
			int imageIndex;
            vector<int>  matchIndexList;
            int trackNumber;
            int visited;
   };

}
