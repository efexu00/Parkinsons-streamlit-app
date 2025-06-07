import joblib
import numpy as np

def load_models():
    xgb = joblib.load("models/xgb_model.pkl")
    rf = joblib.load("models/rf_model.pkl")
    return {"XGBoost": xgb, "Random Forest": rf}

def predict_single(model, features):
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0].max()
    return pred, prob

def predict_batch(model, df):
    preds = model.predict(df)
    probs = model.predict_proba(df).max(axis=1)
    df["Prediction"] = preds
    df["Confidence"] = probs
    return df


