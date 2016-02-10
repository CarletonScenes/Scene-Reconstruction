#Makes a pointcloud of colored points shaped like a simple chapel
from random import randint
import sys

def makeFace(lst,x1, y1, z1, x2, y2, z2, factor):
    xdif = x2-x1
    ydif = y2-y1
    zdif = z2-z1
    x,y,z = x1,y1,z1
    r, g, b = randint(0,254), randint(0,254), randint(0,254)
    if xdif == 0:
        for y in range(factor*y1,factor*(y2+1)-(factor-1)):
            for z in range(factor*z1, factor*(z2+1)-(factor-1)):
                lst.append([x,y/float(factor),z/float(factor),r,g,b])
    elif ydif == 0:
        for x in range(factor*x1,factor*(x2+1)-(factor-1)):
            for z in range(factor*z1, factor*(z2+1)-(factor-1)):
                lst.append([x/float(factor),y,z/float(factor),r,g,b])
    else:
        for y in range(factor*y1,factor*(y2+1)-(factor-1)):
            for x in range(factor*x1, factor*(x2+1)-(factor-1)):
                lst.append([x/float(factor),y/float(factor),z,r,g,b])

def makeChapel(factor, percent):
    f = open("blurredchapel "+str(factor)+" "+str(percent)+".ply","w")
    fileStrings = []
    points = []

    makeFace(points,0,0,0,12,0,4,factor)
    makeFace(points,0,8,0,12,8,4,factor)
    makeFace(points,0,0,0,0,8,4,factor)
    makeFace(points,12,0,0,12,8,4,factor)
    makeFace(points,0,0,4,8,8,4,factor)
    makeFace(points,8,0,4,12,2,4,factor)
    makeFace(points,8,6,4,12,8,4,factor)
    makeFace(points,8,6,4,12,6,10,factor)
    makeFace(points,12,2,4,12,6,10,factor)
    makeFace(points,8,2,4,12,2,10,factor)
    makeFace(points,8,2,4,8,6,10,factor)
    makeFace(points,8,2,10,12,6,10,factor)
    
    points2 = []
    
    for i in range(len(points)):
        x, y, z = points[i][0], points[i][1], points[i][2]
        match = False
#        print(points[i])
        for j in range(i+1,len(points)):
            x1, y1, z1 = points[j][0], points[j][1], points[j][2]
            if x==x1 and y==y1 and z==z1:
                
                match = True
        if not match:
            if randint(0,100)>percent:
                points2.append(points[i])
    
    numPoints = len(points2)
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write("element vertex "+str(numPoints)+"\n")
    f.write("property float32 x\n")
    f.write("property float32 y\n")
    f.write("property float32 z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")
        
    for line in points2:
        string = ""
        for i in range(len(line)):
            if i<3:
                string = string+" "+str(randint(0,10)/50.0+line[i])
            else:
                string = string+" "+str(line[i])
        f.write(string+"\n")

factor = int(sys.argv[1])
percent = int(sys.argv[2])
makeChapel(factor, percent)