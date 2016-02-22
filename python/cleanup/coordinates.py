''' Writes helpful coordinate system '''
from utils.ply_file import PlyFile

def main():
	x, y, z = [], [], []
	increment = 0.1
	depth = 20
	iterations = int(depth / increment)
	for i in range(iterations):
		p = increment * i 
		x.append((p, 0, 0))
		y.append((0, p, 0))
		z.append((0, 0, p))
	points = x + y + z

	outfile = PlyFile()
	outfile.emitPoints(points)
	outfile.save(open('coords.ply', 'w'))

if __name__ == '__main__':
	main()