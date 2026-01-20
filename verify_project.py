import joblib
import os
import numpy as np

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'house_price_model.pkl')

def verify_system():
    print("--- System Verification ---")
    
    # 1. Check Model Existence
    if not os.path.exists(MODEL_PATH):
        print(f"FAILED: Model not found at {MODEL_PATH}")
        return
    print(f"SUCCESS: Model found at {MODEL_PATH}")

    # 2. Load Model
    try:
        model = joblib.load(MODEL_PATH)
        print("SUCCESS: Model loaded with joblib")
    except Exception as e:
        print(f"FAILED: Could not load model. Error: {e}")
        return

    # 3. Test Prediction
    # Features: OverallQual, GrLivArea, TotalBsmtSF, GarageCars, FullBath, YearBuilt
    test_input = [7, 2000, 1000, 2, 2, 2005]
    print(f"Testing prediction with input: {test_input}")
    
    try:
        # Reshape to 2D array as required by sklearn models
        prediction = model.predict([np.array(test_input)])
        print(f"SUCCESS: Prediction generated: ${prediction[0]:,.2f}")
    except Exception as e:
        print(f"FAILED: Prediction error. Error: {e}")
        return

    print("\nSystem Verification Completed Successfully.")
    print("Ready for Deployment.")

if __name__ == "__main__":
    verify_system()
