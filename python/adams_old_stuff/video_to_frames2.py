# from https://stackoverflow.com/questions/18954889/how-to-process-images-of-a-video-frame-by-frame-in-video-streaming-using-opencv

import cv2.cv as cv
videowriter = cv.CreateVideoWriter( filename, fourcc, fps, frameSize)


for frame in ' number of frames:
    ## probs don't need it between EVERY frame, so say twice-four times a second
    cv.WriteFrame( videowriter, frame )