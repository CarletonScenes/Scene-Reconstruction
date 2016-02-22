import sys
import os
import argparse
import utils.triangulate as triangulate
import utils.CVFuncs as CVFuncs
import utils.triangulateManualPoints as triangulateManual
from utils import Image

ACCEPTED_FILETYPES = ['jpg', 'png', 'jpeg']


def addPostToPath(path, post):
    base = os.path.basename(path)
    dirname = os.path.dirname(path)
    base = base.split(".")[0] + "-" + str(post) + "." + base.split(".")[1]

    return dirname + base


def print_help():
    print """Welcome to do_comps.py!
        To run this program, you'll need to select one of the
        modes below and perhaps provide more input.

        modes:
            python do_comps.py detect -i img1.jpg [-i img2.jpg ...] [-f input_folder] [-o output.jpg]
            python do_comps.py match -i img1.jpg -i img2.jpg [-i img3.jpg ...] [-f input_folder] [-o output.jpg]
            python do_comps.py triangulate -i img1.jpg -i img2.jpg [-i img3.jpg ...] [-f input_folder]
                                --scene_output scene.ply [--projection_output projection.ply]
    """


def main(args):
    if len(args) < 2:
        print_help()
        exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', default=None, type=str)
    parser.add_argument('manual_identifier', default=None, nargs='?', type=str)
    parser.add_argument('-i', default=[], action='append', nargs='?', type=str)
    parser.add_argument('-f', default=None, type=str)
    parser.add_argument('-o', default='output.jpg', type=str)
    parser.add_argument('--scene_output', default=None, type=argparse.FileType('w'))
    parser.add_argument('--projection_output', default=None, type=argparse.FileType('w'))
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--naive', action='store_true')

    args = parser.parse_args(args[1:])

    # current_dir = os.path.dirname(os.path.realpath(__file__))
    current_dir = os.getcwd()
    if args.f:
        files = filter(lambda x: any([i in x.lower() for i in ACCEPTED_FILETYPES]), os.listdir(args.f))
        files = map(lambda x: os.path.join(args.f, x), files)
        args.i += files

    args.i = map(lambda x: os.path.join(current_dir, x), args.i)

    mode = args.mode

    if mode == 'detect':
        if not args.silent:
            print 'Detecting images: {}'.format(", ".join(args.i))
            print 'Outputting to: {}'.format(args.o)

        # if there is more than one output image
        if len(args.i) > 1:
            for x in range(len(args.i)):
                image = Image(args.i[x])
                image.detect_features()
                output = image.draw_keypoints(addPostToPath(args.o, x), orientation=True, gray=True)

        else:
            image = Image(args.i[0])
            image.detect_features()
            output = image.draw_keypoints(args.o, orientation=True, gray=True)

    elif mode == 'match':
        if not args.silent:
            print 'Matching images: {}'.format(", ".join(args.i))
            print 'Outputting to: {}'.format(args.o)

        imList = []
        for imageLocation in args.i:
            image1 = Image(imageLocation)
            image1.detect_features()
            imList.append(image1)

        for x in range(0, len(imList)):
            for y in range(x + 1, len(imList)):
                points1, points2, matches = CVFuncs.findMatchesKnn(imList[x], imList[y], filter=True, ratio=True)
                # if there is more than one output image
                if len(imList) > 2:
                    CVFuncs.drawMatches(imList[x], imList[y], matches, addPostToPath(args.o, str(x) + "-" + str(y)))
                else:
                    CVFuncs.drawMatches(imList[x], imList[y], matches, args.o)

    elif mode == 'triangulate':
        if not args.silent:
            print 'Triangulating images: {}'.format(args.i)
            if args.scene_output:
                print 'Outputting scene to: {}'.format(args.scene_output)
            if args.projection_output:
                print 'Outputting projections to: {}'.format(args.projection_output)
        triangulate.triangulateFromImages(args.i,
                                          scene_file=args.scene_output,
                                          projections_file=args.projection_output,
                                          silent=args.silent,
                                          naive=args.naive)

    elif mode == 'manual_pts':
        manual_location = args.manual_identifier

        triangulateManual.triangulateManualAndOutput(args.i[0], args.i[1], manual_location,
                                                     output_file=args.scene_output, projections_file=args.projection_output)


if __name__ == '__main__':
    main(sys.argv)
