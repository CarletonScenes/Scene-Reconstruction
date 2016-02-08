import os
import cv2
import numpy

from image import Image
from image_pair import ImagePair

images = []
count = 0

# Open all images and detect keypoints
current_dir = os.path.dirname(os.path.realpath(__file__))
for f in os.listdir(os.path.join(current_dir, "photos")):
    img = Image(os.path.join(current_dir, "photos/{}".format(f)))
    img.detect_features()
    images.append(img)
    count += 1

# Match all pairs of images and sort
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
all_pairs = []  # [ImagePair]
for i in range(count):
    for j in range(i + 1, count):
        imgA = images[i]
        imgB = images[j]

        matches = bf.match(imgA.descs, imgB.descs)

        pair = ImagePair(imgA, imgB, matches)
        all_pairs.append(pair)

all_pairs.sort(key=lambda x: x.num_matches, reverse=True)

import pprint
pprint.pprint(all_pairs)

# print img1.descs, img2.descs
