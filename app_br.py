'''
The code transforms complex machine learning outputs into an easy-to-use interface, using the 
models we built in the code Model_prep2
'''


import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    h1 {
        color: #4CAF50;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    h2, h3, .stSelectbox, .stNumberInput {
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        font-size: 16px;
        font-family: 'Arial', sans-serif;
    }
    .result-box {
        border: 2px solid #4CAF50;
        padding: 15px;
        border-radius: 5px;
        background-color: #eaf7ea;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and scaler
model_dir = r"C:\\Users\\brian\\OneDrive\\Escritorio\\Lambton\\Semester2\\Toxicology\\Saved_Models"
models = {
    'Random Forest': joblib.load(os.path.join(model_dir, 'RandomForest_model.pkl')),
    'Gradient Boosting': joblib.load(os.path.join(model_dir, 'GradientBoosting_model.pkl')),
    'SVM': joblib.load(os.path.join(model_dir, 'SVM_model.pkl')),
    'Logistic Regression': joblib.load(os.path.join(model_dir, 'LogisticRegression_model.pkl')),
}
scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))

# Initialize session state for prediction history
if "history" not in st.session_state:
    st.session_state["history"] = []

# Streamlit UI
st.title("⚗️ Toxicity Prediction")
st.write("Enter the values for the following features to predict the toxicity level:")

# Input fields for features
st.write("### Input Features")
mw = st.number_input("Molecular Weight (mw)", value=20.0, step=1.0)
polararea = st.number_input("Polar Area (polararea)", value=12.0, step=1.0)
complexity = st.number_input("Complexity (complexity)", value=2.0, step=0.1)
xlogp = st.number_input("XLogP (xlogp)", value=-10.0, step=1.0)
hbonddonor = st.number_input("H-Bond Donor (hbonddonor)", value=0.0, step=1.0)
hbondacc = st.number_input("H-Bond Acceptor (hbondacc)", value=0.0, step=1.0)
rotbonds = st.number_input("Rotatable Bonds (rotbonds)", value=0.0, step=1.0)
heavycnt = st.number_input("Heavy Atom Count (heavycnt)", value=0.0, step=1.0)
exactmass = st.number_input("Exact Mass (exactmass)", value=0.0, step=1.0)
monoisotopicmass = st.number_input("Monoisotopic Mass (monoisotopicmass)", value=0.0, step=1.0)

# Validation for input values
if mw <= 0 or polararea <= 0 or complexity <= 0 or exactmass <= 0 or monoisotopicmass <= 0:
    st.error("All numeric input fields must have positive values!")
else:
    # Select the model to use
    st.write("### Select Prediction Model")
    selected_model = st.selectbox("Choose a model for prediction:", list(models.keys()))

    # Predict button
    if st.button("Predict Toxicity Level"):
        # Prepare the input
        features = np.array([[mw, polararea, complexity, xlogp, hbonddonor, hbondacc, rotbonds, heavycnt, exactmass, monoisotopicmass]])
        scaled_features = scaler.transform(features)
        
        # Predict using the selected model
        model = models[selected_model]
        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]

        # Add prediction to history
        st.session_state["history"].append({
            "Model": selected_model,
            "Prediction": int(prediction),
            "Probabilities": probabilities.tolist(),
            "Features": {
                "mw": mw,
                "polararea": polararea,
                "complexity": complexity,
                "xlogp": xlogp,
                "hbonddonor": hbonddonor,
                "hbondacc": hbondacc,
                "rotbonds": rotbonds,
                "heavycnt": heavycnt,
                "exactmass": exactmass,
                "monoisotopicmass": monoisotopicmass
            }
        })

        # Display results in a styled box
        st.markdown(f"""
            <div class="result-box">
                <h3>Prediction using {selected_model}:</h3>
                <p><b>Predicted Toxicity Level:</b> {int(prediction)}</p>
                <h4>Probabilities for each level:</h4>
                <ul>
                    <li><b>Non-toxic:</b> {probabilities[0]:.2f}</li>
                    <li><b>Mildly Toxic:</b> {probabilities[1]:.2f}</li>
                    <li><b>Moderately Toxic:</b> {probabilities[2]:.2f}</li>
                    <li><b>Highly Toxic:</b> {probabilities[3]:.2f}</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        # Plot probabilities
        st.write("### Probability Distribution")
        labels = ["Non-toxic", "Mildly Toxic", "Moderately Toxic", "Highly Toxic"]
        plt.bar(labels, probabilities, color=['green', 'yellow', 'orange', 'red'])
        plt.xlabel("Toxicity Level")
        plt.ylabel("Probability")
        plt.title("Probability Distribution for Toxicity Levels")
        st.pyplot(plt)

        # Explain the model
        st.write("### Model Explanation")
        if selected_model == "Random Forest":
            st.write("Random Forest uses an ensemble of decision trees to make predictions based on majority voting.")
        elif selected_model == "Gradient Boosting":
            st.write("Gradient Boosting builds an additive model in a forward stage-wise fashion to optimize prediction accuracy.")
        elif selected_model == "SVM":
            st.write("Support Vector Machine (SVM) finds the optimal hyperplane to classify data points into categories.")
        elif selected_model == "Logistic Regression":
            st.write("Logistic Regression predicts the probabilities of categorical outcomes using a linear decision boundary.")

# Display prediction history
st.write("### Prediction History")
if st.session_state["history"]:
    history_df = pd.DataFrame(st.session_state["history"])
    st.write(history_df)
    st.download_button("Download History as CSV", data=history_df.to_csv(index=False), file_name="prediction_history.csv")
else:
    st.write("No predictions yet.")

