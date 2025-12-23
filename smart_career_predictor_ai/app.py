from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        name = request.form['name']
        age = int(request.form['age'])
        score = float(request.form['score'])
        prog = int(request.form['prog'])
        comm = int(request.form['comm'])
        problem = int(request.form['problem'])

        data = np.array([[age, score, prog, comm, problem]])
        prediction = model.predict(data)
        return render_template('result.html', name=name, career=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)