from flask import Flask, render_template
app = Flask(__name__, static_url_path='')


@app.route("/")
def hello():
    return render_template("home.html")


def submit_job():
    pass


@app.route("/stream.ply")
def view_results():
    return app.send_static_file('blurredchapel.ply')

if __name__ == "__main__":
    app.run(debug=True)
