import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
#model = joblib.load('C:/Users/khair/Downloads/calories.h5py') 
#model = pickle.load(open("model.pkl", "rb")) # Load the machine learning model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index_page():
    return render_template('fitness_form.html')

@app.route('/predict_calories', methods=['POST'])
def predict_calories():
    gender = int(request.form.get('gender'))
    age = int(request.form.get('age'))
    height = float(request.form.get('height'))
    weight = float(request.form.get('weight'))
    duration = int(request.form.get('duration'))
    heart_rate = int(request.form.get('heartRate'))
    body_temp = float(request.form.get('bodyTemp'))
    input_data = np.array([[gender, age, height, weight, duration, heart_rate, body_temp]])    
    predicted_calories = model.predict(input_data)
    print(predicted_calories)
    #jsonify({'predicted_calories': predicted_calories.tolist()})
    return render_template('result.html',data=predicted_calories[0])


if __name__ == '__main__':
    app.run(debug=True)
