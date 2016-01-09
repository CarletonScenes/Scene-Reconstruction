import cv2
import exifread

class Image:
    def __init__(self, filepath):
        self.fname = filepath
        self.img = cv2.imread(filepath)
        # self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        with open(filepath, 'r') as f:
            self.tags = exifread.process_file(f)

        self.focal_length_num = self.tags['EXIF FocalLength'].values[0].num
        self.focal_length_den = self.tags['EXIF FocalLength'].values[0].den

        self.kps = None
        self.descs = None

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
