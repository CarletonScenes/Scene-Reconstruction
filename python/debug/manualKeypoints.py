from images import *
import sys
from random import randint

#commandline arguments:
#imagefile1 imagefile2 outputtextfile

#requires sudo apt-get install python-imaging-tk

def highlight(image, point, color):
	for x in range(point[0]-3, point[0]+3):
		for y in range(point[1]-3, point[1]+3):
			if 0 <= x and x < image.getWidth():
				image.setPixel2D(x,y,color)

	image.redraw()


fileName1, fileName2, fileName3 = sys.argv[1], sys.argv[2], sys.argv[3]


image1, image2 = FileImage(fileName1), FileImage(fileName2)
#win = ImageWin()

newImage = EmptyImage(image1.getWidth()+image2.getWidth(), max(image1.getHeight(),image2.getHeight()))

points1 = []
points2 = []

for x in range(image1.getWidth()):
	for y in range(image1.getHeight()):
		newImage.setPixel2D(x,y,image1.getPixel(x,y))

for x in range(image2.getWidth()):
	for y in range(image2.getHeight()):
		newImage.setPixel2D(x+image1.getWidth(),y,image2.getPixel(x,y))

newImage.show()

while len(points1) < 50:
	p1 = newImage.getMouse2D()
	randColor = (randint(0,255),randint(0,255),randint(0,255))

	if p1[0]<image1.getWidth(): # we're in image 1
		highlight(newImage, p1,randColor)
		p2 = newImage.getMouse2D()
		highlight(newImage, p2,randColor)
		points1.append(p1)
		points2.append((p2[0]-image1.getWidth(),p2[1]))

	else:
		highlight(newImage, p1,randColor)
		p2 = newImage.getMouse2D()
		highlight(newImage, p2,randColor)
		points1.append(p2)
		points2.append((p1[0]-image1.getWidth(),p1[1]))

outFile = file(fileName3,"w")

for point1, point2 in zip(points1,points2):
	outFile.write(str(point1[0])+","+str(point1[1])+","+str(point2[0])+","+str(point2[1])+"\n")

for x in range(image1.getWidth()):
	for y in range(image1.getHeight()):
		image1.setPixel2D(x,y,newImage.getPixel(x,y))

for x in range(image2.getWidth()):
	for y in range(image2.getHeight()):
		image2.setPixel2D(x,y,newImage.getPixel(x+image1.getWidth(),y))
f1 = fileName1.split(".")[0]+"-pts"
f2 = fileName2.split(".")[0]+"-pts"

image1.save(fname=None,type='jpg')
image2.save(fname=None,type='jpg')
