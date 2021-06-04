#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request## render_template redirect to the home page in index.html
import pickle
import cv2
import os

MYDIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(MYDIR, "/static/uploads")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__,static_url_path="/static") ## to initialize the flask

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = pickle.load(open('brain_tumor.pkl', 'rb'))

# define from where the user inout is getting
@app.route('/')
def home():
    return render_template('index.html')

# the user input is fed to the model.py to get the predicted value and return the result
@app.route('/predict',methods=['POST'])
def predict():
    '''    For rendering results on HTML GUI ''' 
    file = request.files['search1']
    file_path = "No file"
    if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(file_path)
          
            output = get_result(file_path)
            print("------------------------")
            print(output)
    # display the result in same html page
    return render_template('index.html', prediction_text='cancer is {}'.format(output))


def get_result(file_path):
    '''get the file path and resize the image after that predict using the image'''
    try:
        img = cv2.imread(file_path)
        img = cv2.resize(img,(224,224))
        img = np.expand_dims(img,axis=0)    
        print(model.predict([img]))
        if model.predict([img])[0][0] == 1:
            output = "present"
        else:
            output = "not present"
        return output
    except Exception as e:
        print(str(e))
    

def allowed_file(filename):       
     # ''' check whether the read image belongs to the allowed list'''   
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == "__main__":
    app.run(host="127.0.0.1",port=5000,threaded=False)

