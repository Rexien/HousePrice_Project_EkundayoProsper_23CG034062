from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'house_price_model.pkl')
try:
    model = joblib.load(MODEL_PATH)
except:
    model = None
    print("Error: Model not found. Please train the model first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return render_template('index.html', prediction_text="Error: Model not loaded.")

    try:
        # Get values from form
        features = [
            float(request.form['OverallQual']),
            float(request.form['GrLivArea']),
            float(request.form['TotalBsmtSF']),
            float(request.form['GarageCars']),
            float(request.form['FullBath']),
            float(request.form['YearBuilt'])
        ]
        
        # Reshape for prediction
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        
        output = round(prediction[0], 2)
        
        return render_template('index.html', prediction_text=f'Estimated House Price: ${output:,.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error in prediction: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
