

from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load models
def load_models():
    try:
        nb_model = pickle.load(open('naive_bayes_model.pkl', 'rb'))
        rf_model = pickle.load(open('random_forest_model.pkl', 'rb'))
    except Exception as e:
        print(f"Error loading models: {e}")
        raise
    return nb_model, rf_model

nb_model, rf_model = load_models()

# Crop and fertilizer mappings
crop_dict = { 
    1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut', 
    6: 'papaya', 7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon', 
    11: 'grapes', 12: 'mango', 13: 'banana', 14: 'pomegranate', 
    15: 'lentil', 16: 'blackgram', 17: 'mungbean', 18: 'mothbeans', 
    19: 'pigeonpeas', 20: 'kidneybeans', 21: 'chickpea', 22: 'coffee'
}

fertilizer_dict = {
    0: 'Urea', 1: 'DAP', 2: 'Fourteen-Thirty Five-Fourteen', 
    3: 'Twenty Eight-Twenty Eight', 4: 'Seventeen-Seventeen-Seventeen', 
    5: 'Twenty-Twenty', 6: 'Ten-Twenty Six-Twenty Six'
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        try:
            # Extracting input data from the form
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            # Preprocess the data and predict
            crop_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            crop_prediction_num = nb_model.predict(crop_features)[0]
            crop_name = crop_dict.get(crop_prediction_num, "Unknown Crop")

            fertilizer_features = np.array([[N, P, K]])
            fertilizer_prediction = rf_model.predict(fertilizer_features)[0]
            fertilizer_name = fertilizer_dict.get(fertilizer_prediction, "Unknown Fertilizer")

            return render_template('prediction.html', crop_prediction=crop_name, fertilizer_prediction=fertilizer_name)
        except Exception as e:
            return render_template('prediction.html', error=str(e))

    return render_template('prediction.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
