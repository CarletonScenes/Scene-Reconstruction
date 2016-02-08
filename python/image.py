import numpy as np
import cv2
import exifread


class Image:

    def __init__(self, filepath):
        self.fname = filepath
        self.img = cv2.imread(filepath)
        # self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.height = self.img.shape[0]
        self.width = self.img.shape[1]

        try:
            with open(filepath, 'r') as f:
                self.tags = exifread.process_file(f)
            self.focal_length_num = self.tags['EXIF FocalLength'].values[0].num
            self.focal_length_den = self.tags['EXIF FocalLength'].values[0].den
        except:
            self.tags = {}
            self.focal_length_num = 1
            self.focal_length_den = 1

        self.kps = None
        self.descs = None

        focal_length = self.focal_length_num / self.focal_length_den
        cx = float(self.width) / 2
        cy = float(self.height) / 2
        self.k = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ])

        self.k_inv = np.linalg.inv(self.k)

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
