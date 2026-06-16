# =============================================================
# FILE: api/main.py
# PURPOSE: This is the API (Application Programming Interface)
#          It is NOT a database. It is a BRIDGE/MIDDLEMAN that
#          connects the Website (frontend) with the ML Model.
#
# HOW IT WORKS:
#   Website (app.js)  -->  FastAPI (this file)  -->  ML Model (.pkl)
#   User types text   -->  API receives text    -->  Model predicts
#   Website shows     <--  API sends result     <--  Phishing/Legit
#
# Think of it like a WAITER in a restaurant:
#   - You (website) give your order to the waiter (API)
#   - Waiter brings it to the kitchen (ML model)
#   - Kitchen cooks it and waiter brings result back to you
# =============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Create the FastAPI application (this IS the API/bridge)
app = FastAPI(title="Phishing Detection API")

# CORS = Cross-Origin Resource Sharing
# This allows the Website (running on port 5500) to talk to
# this API (running on port 8000). Without this, the browser
# would BLOCK the connection for security reasons.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all websites to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models and Vectorizers placeholders
models = {}

def load_model_suite(prefix):
    """
    This function loads the saved ML model files (.pkl) from the models folder.
    These .pkl files were created during model training in Training_Notebook.ipynb.
    - vectorizer : converts text into numbers (TF-IDF)
    - mask       : the BHHO feature selection mask (which features to keep)
    - model      : the trained Random Forest classifier
    """
    try:
        base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        vectorizer = joblib.load(os.path.join(base_path, f'{prefix}_vectorizer.pkl'))  # TF-IDF
        mask = joblib.load(os.path.join(base_path, f'{prefix}_feature_mask.pkl'))      # BHHO mask
        model = joblib.load(os.path.join(base_path, f'{prefix}_model.pkl'))            # Random Forest
        return {'vectorizer': vectorizer, 'mask': mask, 'model': model}
    except Exception as e:
        print(f"Warning: Could not load {prefix} models. Make sure you run the training pipeline first. Error: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    # Load all models on startup
    models['email'] = load_model_suite('email')
    models['sms'] = load_model_suite('sms')
    models['url'] = load_model_suite('url')

class TextInput(BaseModel):
    text: str

def predict(input_text: str, prefix: str):
    """
    This is the MAIN function that connects the website input to the ML model.
    It runs the full ML pipeline every time the user clicks 'Analyze'.

    PIPELINE (step by step):
    Website sends text --> Step 1 TF-IDF --> Step 2 BHHO --> Step 3 Random Forest --> Result
    """
    suite = models.get(prefix)
    if not suite:
        raise HTTPException(status_code=503, detail=f"{prefix.capitalize()} model not loaded. Train models first.")

    # ---------------------------------------------------------------
    # STEP 1: FEATURE EXTRACTION using TF-IDF
    # Convert the user's raw text into numbers that the ML model understands
    # Example: "click here verify account" --> [0.0, 0.8, 0.7, 0.6, 0.0, ...]
    # ---------------------------------------------------------------
    features = suite['vectorizer'].transform([input_text]).toarray()

    # ---------------------------------------------------------------
    # STEP 2: FEATURE SELECTION using BHHO Mask
    # BHHO already selected the best features during training.
    # Here we apply that mask to keep only the important features.
    # Example: from 5000 features, BHHO keeps only 300 important ones
    # ---------------------------------------------------------------
    selected_indices = np.where(suite['mask'] == 1)[0]
    features_selected = features[:, selected_indices]

    # ---------------------------------------------------------------
    # STEP 3: PREDICTION using Random Forest
    # The trained Random Forest model reads the selected features
    # and decides: is this Phishing (1) or Legitimate (0)?
    # ---------------------------------------------------------------
    prediction = suite['model'].predict(features_selected)[0]

    # ---------------------------------------------------------------
    # STEP 4: CONFIDENCE SCORE
    # predict_proba() gives the probability of each class.
    # Example: [0.09, 0.91] means 91% confident it is Phishing.
    # We take the highest probability as the AI Confidence score.
    # ---------------------------------------------------------------
    try:
        proba = suite['model'].predict_proba(features_selected)[0]
        confidence = float(max(proba))  # take the highest probability
    except:
        confidence = 1.0  # fallback if model does not support proba

    # ---------------------------------------------------------------
    # STEP 5: EXTRACT FLAGGED KEYWORDS (only if phishing detected)
    # Find which suspicious words from the user's input were
    # among the BHHO-selected important features.
    # These keywords are shown on the website as red tags.
    # ---------------------------------------------------------------
    flagged_keywords = []
    if prediction == 1:
        try:
            feature_names = suite['vectorizer'].get_feature_names_out()
            non_zero_input_indices = np.where(features[0] > 0)[0]          # words user typed
            flagged_indices = np.intersect1d(non_zero_input_indices, selected_indices)  # overlap with BHHO
            flagged_keywords = [feature_names[i] for i in flagged_indices]
            flagged_keywords = flagged_keywords[:5]  # show max 5 keywords
        except Exception as e:
            print(f"Keyword extraction error: {e}")

    # ---------------------------------------------------------------
    # STEP 6: SEND RESULT BACK TO WEBSITE (as JSON)
    # This JSON is received by app.js in the website
    # and displayed to the user on screen.
    # ---------------------------------------------------------------
    return {
        "prediction": "Phishing" if prediction == 1 else "Legitimate",  # shown as result text
        "confidence": round(confidence * 100, 2),                        # shown as % bar
        "is_phishing": bool(prediction == 1),                            # used for red/green color
        "flagged_keywords": flagged_keywords                              # shown as keyword tags
    }

# =============================================================
# API ENDPOINTS - These are the URLs that the website calls
# The website (app.js) sends a POST request to these addresses:
#   Email tab --> POST http://127.0.0.1:8000/predict/email
#   SMS tab   --> POST http://127.0.0.1:8000/predict/sms
#   URL tab   --> POST http://127.0.0.1:8000/predict/url
# =============================================================

@app.post("/predict/email")  # website calls this for email scanning
async def predict_email(request: TextInput):
    return predict(request.text, 'email')

@app.post("/predict/sms")    # website calls this for SMS scanning
async def predict_sms(request: TextInput):
    return predict(request.text, 'sms')

@app.post("/predict/url")    # website calls this for URL scanning
async def predict_url(request: TextInput):
    return predict(request.text, 'url')

@app.get("/")
async def root():
    return {"message": "Phishing Detection API is running."}
