# coding=utf-8
import os
import numpy as np
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer
import cv2
import base64
from PIL import Image
from io import BytesIO
from detector import *
def img_decode(f):
    _, a_numpy = cv2.imencode('.jpg', f)
    a = a_numpy.tobytes()
    encoded = base64.encodebytes(a)
    image_encoded = encoded.decode('utf-8')
    return image_encoded

app = Flask(__name__)
def get_file_path_and_save(request):
    # Get the file from post request
    f = request.files['file']
    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'imgs', secure_filename(f.filename))
    f.save(file_path)
    return file_path

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index_2.html')

@app.route('/predictResNet_tt', methods=['GET', 'POST'])
def predictResNet_tt():
    if request.method == 'POST':
        file = request.files['file']
        image_file = file.read()
        img = cv2.imdecode(np.frombuffer(image_file, dtype=np.uint8), -1)
        img,num_box,t = detect(img,'tt')
        img = img_decode(img)
        data = {'image':img,'box':num_box,'time':t}
        return data
    return None
@app.route('/predictResNet_ctw', methods=['GET', 'POST'])
def predictResNet_ctw():
    if request.method == 'POST':
        file = request.files['file']
        image_file = file.read()
        img = cv2.imdecode(np.frombuffer(image_file, dtype=np.uint8), -1)
        img,num_box,t = detect(img,'ctw')
        img = img_decode(img)
        data = {'image':img,'box':num_box,'time':t}
        return data
    return None

@app.route('/retrained', methods=['GET'])
def index_2():
    # Main page
    return render_template('index.html')

@app.route('/retrained/predictResNet50', methods=['GET', 'POST'])
def predictResNet50():
    if request.method == 'POST':
        file = request.files['file']
        image_file = file.read()
        img = cv2.imdecode(np.frombuffer(image_file, dtype=np.uint8), -1)
        img,num_box,t = detect(img,'50')
        img = img_decode(img)
        data = {'image':img,'box':num_box,'time':t}
        return data
    return None
@app.route('/retrained/predictResNet18', methods=['GET', 'POST'])
def predictResNet18():
    if request.method == 'POST':
        file = request.files['file']
        image_file = file.read()
        img = cv2.imdecode(np.frombuffer(image_file, dtype=np.uint8), -1)
        img,num_box,t = detect(img,'18')
        img = img_decode(img)
        data = {'image':img,'box':num_box,'time':t}
        return data
    return None

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='8080',debug=True)