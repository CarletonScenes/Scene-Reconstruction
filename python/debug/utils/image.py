import numpy as np
import cv2
import exifread
import sys
from KMatrix import KMatrix


class Image:

    def __init__(self, filepath):
        self.fname = filepath
        self.img = cv2.imread(filepath)
        if self.img is None:
            raise IOError("File " + filepath + " could not be read.")
        self.height = self.img.shape[0]
        self.width = self.img.shape[1]

        # Resize to 1000x1000 if too large
<<<<<<< d2e6b2c26c30364f097cc3c00a4cac8a973316d5
        # This was a dumb idea and I'm going to comment it out
        # if self.height > 1000 or self.width > 1000:
        #     self.height = 1000
        #     self.width = 1000
        #     self.img = cv2.resize(self.img, (self.height, self.width))
=======
        if self.height > 1000 or self.width > 1000:
            self.height = 1000
            self.width = 1000
            self.img = cv2.resize(self.img, (self.height, self.width))
>>>>>>> Add image resizing

        try:
            with open(filepath, 'r') as f:
                self.tags = exifread.process_file(f)
            self.focal_length_num = float(self.tags['EXIF FocalLength'].values[0].num)
            self.focal_length_den = self.tags['EXIF FocalLength'].values[0].den
        except:
            self.tags = {}
            self.focal_length_num = 1
            self.focal_length_den = 1

        self.kps = None
        self.descs = None

        # CALCULATE K

        # First, find focal length in mm
        mm_focal_length = self.focal_length_num / self.focal_length_den
        # CCD width is the width of the image sensor (using iphone as default for now)
        ccdWidth = 4.89
        # Calculate focal length in pixels
        self.focalLength = self.width * mm_focal_length / ccdWidth

        center_x = float(self.width) / 2
        center_y = float(self.height) / 2

        self.scale_factor = self.width / ccdWidth  # note: equiv to mm_focal_length / self.focalLength. multiply this by pixels to get mm

        # multiply by pixel coords (x, y, 1) and remain in pixels -- this just translates so image center is origin
        self.naiveK = KMatrix(focalLength=1, principalPoint=(center_x, center_y))

        self.K = KMatrix(focalLength=self.focalLength, principalPoint=(center_x, center_y))

    def detect_features(self):
        sift = cv2.xfeatures2d.SIFT_create()
        (self.kps, self.descs) = sift.detectAndCompute(self.img, None)

    def __repr__(self):
        if not self.kps and not self.descs:
            return "{} - Focal Length: {} / {} - KPs: {} Descs: {}".format(
                self.fname,
                self.focal_length_num,
                self.focal_length_den)
        else:
            return "{} - Focal Length: {} / {} - KPs: {} Descs: {}".format(
                self.fname,
                self.focal_length_num,
                self.focal_length_den,
                len(self.kps),
                len(self.descs))
