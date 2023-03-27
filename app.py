from flask import Flask, render_template, request

from Malaria import Malaria

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html", res="None")


@app.route('/predictdisease', methods=['POST'])
def predictdisease():
    app_root = request.host_url
    file = request.files['img']
    filename = file.filename
    #file_content = file.read()
    save_path = "{0}{1}".format("static/images/Malaria Cells/single_prediction/", filename)
    url_save_path = "{0}{1}".format("images/Malaria Cells/single_prediction/", filename)
    file.save(save_path)

    malaria = Malaria()
    res = malaria.predict(filename)
    return render_template("index.html", res=res, url_save_path = url_save_path, app_root=app_root)

