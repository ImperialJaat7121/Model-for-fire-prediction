import pickle
from flask import Flask,request, jsonify, render_template
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application

ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def greet():
    return render_template('greet.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_scale_data = scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_scale_data)

        return render_template('prediction_page.html',prediction=result[0])

    if request.method == 'GET':
        return render_template('prediction_page.html')
       



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)