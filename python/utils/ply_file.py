import StringIO


class PlyFile:

    def __init__(self, color=False):
        self.color = color
        self.output_str = StringIO.StringIO()
        self.points = []

    def write(self, input):
        pass

    def writeline(self, line):
        return self.output_str.write("{}\n".format(line))

    # Writes a standard ply header with only verticies to a file-like obj
    def emitHeader(self):
        self.writeline("ply")
        self.writeline("format ascii 1.0")
        self.writeline("element vertex {}".format(len(self.points)))
        self.writeline("property float x")
        self.writeline("property float y")
        self.writeline("property float z")
        self.writeline("end_header")

    # # Writes a color header with only verticies to a file-like obj
    # def emitColorHeader(self):
    #     self.writeline("ply")
    #     self.writeline("format ascii 1.0")
    #     self.writeline("element vertex {}".format(pts.shape[0]))
    #     self.writeline("property float x")
    #     self.writeline("property float y")
    #     self.writeline("property float z")
    #     self.writeline("property uchar red")
    #     self.writeline("property uchar green")
    #     self.writeline("property uchar blue")
    #     self.writeline("end_header")

    # Writes some regular 3d points to a given file-like object
    def emitPoints(self, points):
        self.points += points

    def save(self, file):
        if self.color:
            # self.emitColorHeader()
            pass
        else:
            self.emitHeader()

        for point in self.points:
            self.writeline("%f %f %f" % (point[0], point[1], point[2]))

        file.write(self.output_str.getvalue())
