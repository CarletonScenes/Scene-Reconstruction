import uuid
import json
import os
import utils.triangulate as triangulate
import utils.CVFuncs as CVFuncs
from flask import Flask, render_template, request
from utils import Image
from werkzeug import secure_filename

print triangulate.triangulateFromImages

app = Flask(__name__, static_url_path='/static')
try:
    os.mkdir("/tmp/sceneflask")
except:
    pass
app.config['UPLOAD_FOLDER'] = "/tmp/sceneflask"
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'gif'])

OUTPUT_IMAGE_PATH = os.path.abspath("./static/output_imgs")


def hello():
    return render_template("home.html")


@app.route("/")
def make_upload_form():
    job_id = uuid.uuid4()
    return render_template("submit_job.html", job_id=job_id)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


def save_files(fileDict):
    print fileDict
    filepaths = []
    for fileObj in fileDict.getlist('files'):
        if file and allowed_file(fileObj.filename):
            filename = secure_filename(fileObj.filename)
            newpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            fileObj.save(newpath)
            filepaths.append(newpath)
    return filepaths


@app.route("/detect", methods=["POST"])
def detect_image():
    filepaths = save_files(request.files)
    output_paths = []
    for file in filepaths:
        image = Image(file)
        image.detect_features()
        output_filename = os.path.join(OUTPUT_IMAGE_PATH, str(uuid.uuid4()) + "_kps.jpg")
        output = image.draw_keypoints(output_filename, orientation=True, gray=True)
        output_paths.append(output_filename[output_filename.find("/static"):])

    print output_paths

    return json.dumps({"output_paths": output_paths})


@app.route("/match", methods=["POST"])
def match_images():
    filepaths = save_files(request.files)
    output_paths = []
    imList = []
    for imageLocation in filepaths:
        image1 = Image(imageLocation)
        image1.detect_features()
        imList.append(image1)

    print imList
    for x in range(0, len(imList)):
        for y in range(x + 1, len(imList)):
            points1, points2, matches = CVFuncs.findMatchesKnn(imList[x], imList[y], filter=True, ratio=True)
            print points1
            output_filename = os.path.join(OUTPUT_IMAGE_PATH, str(uuid.uuid4()) + "_match.jpg")
            CVFuncs.drawMatches(imList[x], imList[y], matches, output_filename)
            output_paths.append(output_filename[output_filename.find("/static"):])

    print output_paths

    return json.dumps({"output_paths": output_paths})


@app.route("/triangulate", methods=["POST"])
def triangulate_handler():
    filepaths = save_files(request.files)
    scene_ply_location = os.path.join(OUTPUT_IMAGE_PATH, str(uuid.uuid4()) + "_scene.ply")
    proj_ply_location = os.path.join(OUTPUT_IMAGE_PATH, str(uuid.uuid4()) + "_proj.ply")

    scene_ply_file = open(scene_ply_location, 'w')
    proj_ply_file = open(proj_ply_location, 'w')

    print triangulate
    print triangulate.triangulateFromImages

    triangulate.triangulateFromImages(filepaths,
                                      scene_file=scene_ply_file,
                                      projections_file=proj_ply_file,
                                      silent=True,
                                      cv=True)

    return json.dumps({
        'scene': scene_ply_location[scene_ply_location.find("/static"):],
        'proj': proj_ply_location[proj_ply_location.find("/static"):]
    })


@app.route("/stream.ply")
def view_results():
    return app.send_static_file('manualprojections.ply')

if __name__ == "__main__":
    app.run(debug=True)
