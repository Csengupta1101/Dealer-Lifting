import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import json

app = Flask(__name__)
#loading the model
regmodel = pickle.load(open('regmodel.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    data=[int(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    output=regmodel.predict(final_input)
    output = round(output[0],2)
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)