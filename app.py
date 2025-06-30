from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model/house_price_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['income']),
            float(request.form['age']),
            float(request.form['rooms']),
            float(request.form['bedrooms']),
            float(request.form['population'])
        ]
        prediction = model.predict([np.array(features)])
        price = f"${prediction[0]:,.2f}"
        return render_template('index.html', prediction_text=f"Estimated House Price: {price}")
    except Exception as e:
        return render_template('index.html', prediction_text="Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
