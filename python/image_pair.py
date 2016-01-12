
class ImagePair:
    def __init__(self, imageA, imageB, matches):
        self.imageA = imageA
        self.imageB = imageB
        self.num_matches = len(matches)
        self.matches = matches

    def __repr__(self):
        return "<ImagePair Matches: {} A: {} B: {}".format(
            self.num_matches, 
            self.imageA.fname, 
            self.imageB.fname)
