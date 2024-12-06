import streamlit as st
import joblib
import numpy as np

# Load pre-trained models
model_1 = joblib.load('model_1_toxicity.pkl')  # Toxicity prediction model
model_2 = joblib.load('model_2_toxicity_level.pkl')  # Toxicity level prediction model

# Streamlit app title
st.title("Toxicity Prediction System")

# Input fields for chemical properties
st.header("Input Chemical Properties")
mw = st.number_input("Molecular Weight (mw)", value=0.0)
polararea = st.number_input("Polar Surface Area (polararea)", value=0.0)
complexity = st.number_input("Structural Complexity (complexity)", value=0.0)
xlogp = st.number_input("Hydrophobicity (xlogp)", value=0.0)
heavycnt = st.number_input("Heavy Atom Count (heavycnt)", value=0)
hbonddonor = st.number_input("Hydrogen Bond Donors (hbonddonor)", value=0)
rotbonds = st.number_input("Rotatable Bonds (rotbonds)", value=0)
totalatomstereocnt = st.number_input("Total Atom Stereocenters (totalatomstereocnt)", value=0)
totalbondstereocnt = st.number_input("Total Bond Stereocenters (totalbondstereocnt)", value=0)

# Predict button
if st.button("Predict Toxicity"):
    # Prepare feature array
    features = np.array([mw, polararea, complexity, xlogp, heavycnt, hbonddonor, rotbonds, totalatomstereocnt, totalbondstereocnt]).reshape(1, -1)
    
    # Model 1: Predict if compound is toxic
    is_toxic = model_1.predict(features)[0]
    st.write(f"Toxicity Prediction: {'Toxic' if is_toxic else 'Non-Toxic'}")
    
    # If toxic, predict toxicity level using Model 2
    if is_toxic:
        toxicity_level = model_2.predict(features)[0]
        st.write(f"Toxicity Level: {toxicity_level}")

# Streamlit footer
st.write("Developed for Toxicology Analysis")
