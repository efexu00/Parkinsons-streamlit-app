import streamlit as st# ✅ Load models right after imports
import pandas as pd
import numpy as np
import shap
import pickle
from utils.model_utils import load_models, predict_single, predict_batch
from utils.audio_features import extract_features
from utils.shap_utils import plot_shap_waterfall
from utils.logger import log_upload
import io

# ✅ Load models right after imports
models = load_models()
xgb_model = models["XGBoost"]
rf_model = models["Random Forest"]


EXPECTED_FEATURES = 768  # Update if your model expects a different number

def pad_or_trim_features(features, target_dim=EXPECTED_FEATURES):
    """Pad or trim feature vector to match model input shape."""
    if features.shape[1] < target_dim:
        padding = target_dim - features.shape[1]
        features = np.pad(features, ((0, 0), (0, padding)), mode='constant')
    elif features.shape[1] > target_dim:
        features = features[:, :target_dim]
    return features

# Streamlit config
st.set_page_config(page_title="PD Detection App", layout="wide")
st.title("🧠 Parkinson's Disease Detection from Voice")
st.write("Upload a .wav file to classify Parkinson's presence using AI models.")

# Sidebar Inputs
model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "XGBoost"])
audio_file = st.sidebar.file_uploader("Upload .wav File", type=["wav"])
batch_file = st.sidebar.file_uploader("Upload Batch CSV", type=["csv"])
metadata_file = st.sidebar.file_uploader("Upload Metadata CSV", type=["csv"])

# 🧠 Single Prediction
if audio_file is not None:
    st.audio(audio_file, format="audio/wav")

    try:
        features = extract_features(audio_file)

        models = load_models()
        xgb_model = models["XGBoost"]
        rf_model = models["Random Forest"]

        xgb_pred, xgb_conf = predict_single(xgb_model, features)
        rf_pred, rf_conf = predict_single(rf_model, features)

        st.subheader("🔍 Model Predictions")
        st.write(f"🎯 XGBoost: {xgb_pred} (Confidence: {xgb_conf:.2f})")
        st.write(f"🌲 Random Forest: {rf_pred} (Confidence: {rf_conf:.2f})")

        # Optional: Logging and SHAP
        log_upload("User", audio_file.name, model_choice, xgb_pred)
        shap_fig = plot_shap_waterfall(xgb_model, features)
        st.pyplot(shap_fig)

        # Downloadable Report
        report_df = pd.DataFrame(features, columns=[f"F{i+1}" for i in range(features.shape[1])])
        report_df["Prediction"] = xgb_pred
        report_df["Confidence"] = np.round(xgb_conf * 100, 2)
        st.download_button("📥 Download Prediction Report", report_df.to_csv(index=False).encode(), "prediction_report.csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

# Batch Predictions
if batch_file:
    try:
        # Read uploaded CSV
        df = pd.read_csv(batch_file)

        # Load models (don't pass model_choice unless your function supports it)
        models = load_models()  # Assuming load_models() takes NO arguments
        model = models[model_choice]  # Safely select model by key

        # Make batch predictions
        predictions = predict_batch(model, df)

        # Display results
        st.subheader("📊 Batch Prediction Results")
        st.dataframe(predictions)

        # Download button for results as CSV
        csv = predictions.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Batch Results", csv, "batch_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error in batch processing: {e}")


# Metadata Viewer
if metadata_file:
    try:
        st.subheader("🧾 Patient Metadata")
        metadata_df = pd.read_csv(metadata_file)
        st.dataframe(metadata_df)
    except Exception as e:
        st.error(f"❌ Error loading metadata file: {e}")

