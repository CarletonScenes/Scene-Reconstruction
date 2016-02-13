"""
image.py This module provides a simple interface to create a window,
load an image and experiment with image based algorithms.  Many of
which require pixel-by-pixel manipulation.  This is a educational
module, its not intended to replace the excellent Python Image
Library, in fact it uses PIL.

The module and its interface and some of the code were inspired/copied
by/from John Zelle's graphics.py which serves a similar purpose in the
graphics primitive world.
"""

### TODO
### EmptyImage creates a color image, what if you want gray? Might want
### to just eliminate that
###
### I still really like the 1D vs 2D methods, but I don't love the names
### 1D vs 2D

# Originally written by Brad Miller, at Luther College
# Modifed by Dave Musicant, at Carleton College, based on Brad's version 1.2

import Tkinter
import Image
import ImageTk


# Borrow some ideas from Zelle
# create an invisible global main root for all windows
tk = Tkinter
_imroot = tk.Tk()
_imroot.withdraw()


class ImageWin(tk.Canvas):
    """
    ImageWin:  Make a frame to display one or more images.
    """
    def __init__(self,title,width,height):        
        """
        Create a window with a title, width and height.
        """
        master = tk.Toplevel(_imroot)
        master.protocol("WM_DELETE_WINDOW", self.close)
        tk.Canvas.__init__(self, master, width=width, height=height)
        self.master.title(title)
        self.pack()
        master.resizable(0,0)
        self.foreground = "black"
        self.items = []
        self.mouseX = None
        self.mouseY = None
        self.bind("<Button-1>", self._onClick)
        self.height = height
        self.width = width
        self._mouseCallback = None
        self.trans = None
        _imroot.update()

    def setLocation(self,x,y):
        self.master.geometry(str(self.width) + 'x' + str(self.height) +
                             '+' + str(x) + '+' + str(y))

    def close(self):
        """Close the window"""
        self.master.destroy()
        self.quit()
        _imroot.update()
        
    def getMouse(self):
        """Wait for mouse click and return a tuple with x,y position in screen coordinates after
        the click"""
        self.mouseX = None
        self.mouseY = None
        while self.mouseX == None or self.mouseY == None:
            self.update()
        return ((self.mouseX,self.mouseY))

    def setMouseHandler(self, func):
        self._mouseCallback = func

    def _onClick(self, e):
        self.mouseX = e.x
        self.mouseY = e.y
        if self._mouseCallback:
            self._mouseCallback(Point(e.x, e.y)) 
 

class AbstractImage:
    """
    Create an image.  The image may be created in one of four ways:
    1. From an image file such as gif, jpg, png, ppm  for example: i = image('fname.jpb)
    2. From a list of lists
    3. From another image object
    4. By specifying the height and width to create a blank image.
    """
    imageCache = {} # tk photoimages go here to avoid GC while drawn 
    imageId = 1
    def __init__(self,fname=None,data=[],imobj=None,height=0,width=0):
        """
        An image can be created using any of the following keyword parameters. When image creation is 
        complete the image will be an rgb image.
        fname:  A filename containing an image.  Can be jpg, gif, and others
        data:  a list of lists representing the image.  This might be something you construct by
        reading an asii format ppm file, or an ascii art file and translate into rgb yourself.
        imobj:  Make a copy of another image.
        height:
        width: Create a blank image of a particular height and width.
        """
        if fname:
            self.im = Image.open(fname)
            self.imFileName = fname
            ni = self.im.convert("RGB")
            self.im = ni
        elif data:
            height = len(data)
            width = len(data[0])
            self.im = Image.new("RGB",(width,height))
            for row  in range(height):
                for col in range(width):
                    self.im.putpixel((col,row),data[row][col])
        elif height > 0 and width > 0:
            self.im = Image.new("RGB",(width,height))
        elif imobj:
            self.im = imobj.copy()
            
        self.width,self.height = self.im.size
        self.centerX = self.width/2+3     # +3 accounts for the ~3 pixel border in Tk windows
        self.centerY = self.height/2+3
        self.id = None
        self.pixels = self.im.load()

    def copy(self):
        """Return a copy of this image"""
        newI = AbstractImage(imobj=self.im)
        return newI

    def copyGray(self):
        newI = AbstractImage(imobj=self.im.convert('L'))
        return newI

    def clone(self):
	     """Return a copy of this image"""
	     newI = AbstractImage(imobj=self.im)
	     return newI
        
    def getHeight(self):
        """Return the height of the image"""
        return self.height

    def getWidth(self):
        """Return the width of the image"""
        return self.width

    def getNumPixels(self):
        """Return the number of pixels in the image"""
        return self.height * self.width

    def getPixel(self,x,y):
        """Get a pixel at the given x,y coordinate.  The pixel is returned as an rgb color tuple
        for eaxample foo.getPixel(10,10) --> (10,200,156) """
        return self.pixels[x,y]
        
    def getPixel2D(self,x,y):
        """Get a pixel at the given x,y coordinate.  The pixel is returned as
        an rgb color tuple for example foo.getPixel(10,10) -->
        (10,200,156) """
        return self.pixels[x,y]

    def getPixel1D(self,loc):
        """Get a pixel at the given location, considering the pixels as one
        long 1 dimensional string of numbers.coordinate.  The pixel is
        returned as an rgb color tuple for example
        foo.getPixel(10) --> (10,200,156) """
        return self.pixels[loc % self.width,loc / self.width]
        
    def setPixel2D(self,x,y,color):
        """Set the color of a pixel at position x,y.  The color must be
        specified as an rgb tuple [r,g,b] where the rgb values are
        between 0 and 255."""
        self.pixels[x,y] = color
    
    def setPixel1D(self,loc,color):
        self.pixels[loc % self.width,loc / self.width] = color
        
    def show(self,title='Image',x=None,y=None):
        """Draw this image in the ImageWin window."""
        ig = ImageTk.PhotoImage(self.im)
        self.imageCache[self.imageId] = ig # save a reference else Tk loses it...
        AbstractImage.imageId = AbstractImage.imageId + 1
        self.canvas=ImageWin(title,self.width,self.height)
        if x != None and y != None:
            self.canvas.setLocation(x,y)
        elif x != None:
            raise TypeError, "You specified the x location but not the y"
        self.id = self.canvas.create_image(self.centerX,self.centerY,image=ig)
        _imroot.update()

    def close(self):
        self.canvas.close()

    def redraw(self):
        """Draw this image in the ImageWin window."""
        ig = ImageTk.PhotoImage(self.im)
        self.imageCache[self.imageId] = ig # save a reference else Tk loses it...
        AbstractImage.imageId = AbstractImage.imageId + 1
        self.id = self.canvas.create_image(self.centerX,self.centerY,image=ig)
        _imroot.update()

    def save(self,fname=None,type='jpg'):
        if fname == None:
            fname = self.imFileName
        try:
            self.im.save(fname)            
        except:
            print "Error saving, Could Not open ", fname, " to write."

    def toList(self):
        """
        Convert the image to a List of Lists representation
        """
        res = []
        for i in range(self.height):
            res.append([])
            for j in range(self.width):
                res[i].append(self.pixels[j,i])
        return res

    def getMouse1D(self):
        x, y = self.canvas.getMouse()
        return y * self.width + x

    def getMouse2D(self):
        return self.canvas.getMouse()

class FileImage(AbstractImage):
    def __init__(self,thefile):
        AbstractImage.__init__(self,fname = thefile)

class EmptyImage(AbstractImage):
    def __init__(self,cols,rows):
      	AbstractImage.__init__(self,height = rows, width = cols)

class ListImage(AbstractImage):
	def __init__(self,thelist):
		AbstractImage.__init__(self,data=thelist)

# Example program  Read in an image and calulate the negative.
if __name__ == '__main__':
    oImage = FileImage('dave.jpg')
    oImage.show('orig')
    myImage = oImage.copyGray()
    myImage.show('gray',500,500)
#     #myImage.setPixel2D(0,0,[0,0,0])
#     #myImage.setPixel1D(0,(0,0,0))
#     myImage.setPixel2D(0,0,0)
#     myImage.draw(400,500)
#     myImage.draw(400,500)
    raw_input("Press enter")


#     for row in range(myImage.getHeight()):
#         for col in range(myImage.getWidth()):
#              v = myImage.getPixel(col,row)
#              x = map(lambda x: 255-x, v)
#              myImage.setPixel(col,row,tuple(x))
#     myImage.setPosition(300,300)         
#     myImage.draw(win)
#     win.getMouse()
#     #myImage.save('/Users/bmiller/tmp/testfoo.jpg')
#     #print myImage.toList()
#     win.close()
#     win = ImageWin("window 2",300,300)


