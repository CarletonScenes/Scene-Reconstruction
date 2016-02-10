from flask import Flask, render_template
app = Flask(__name__)


@app.route("/")
def hello():
    return render_template("home.html")


def submit_job():
    pass


@app.route("/job/:job_id")
def view_results(job_id):
    pass

if __name__ == "__main__":
    app.run()
