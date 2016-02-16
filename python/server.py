import uuid
from flask import Flask, render_template
app = Flask(__name__, static_url_path='')


@app.route("/")
def hello():
    return render_template("home.html")


def make_upload_form():
    job_id = uuid.uuid4()

    return render_template("submit_job.html", job_id=job_id)


def submit_job():
    pass


@app.route("/stream.ply")
def view_results():
    return app.send_static_file('manualprojections.ply')

if __name__ == "__main__":
    app.run(debug=True)
