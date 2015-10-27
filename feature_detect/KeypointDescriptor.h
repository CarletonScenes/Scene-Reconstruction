#pragma once
#include <stdio.h>
#include <unistd.h>
#include "Image.h"
#include "OpenCV.h"

namespace SceneComps {
   class KeypointDescriptor: public cv::KeyPoint {
       public:
			KeypointDescriptor(KeyPoint kp) : cv::KeyPoint () {
				// Always init with a keypoint
				pt = kp.pt;
				size = kp.size;
				angle = kp.angle;
				response = kp.response;
				octave	= kp.octave;
				class_id = kp.class_id;
        	};
			Ptr<Image> image;       
   };

}
