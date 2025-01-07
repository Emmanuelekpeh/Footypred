import streamlit as st
import numpy as np
import joblib

# Load Models
st.title("Football Match Prediction App")
st.write("Predict match outcomes such as Over 2.5 goals and match results.")

# Load models
model_match_result = joblib.load("model_result.pkl")
model_o2_5 = joblib.load("model_o2_5.pkl")

# Input Features
st.sidebar.header("Input Features")
home_odds = st.sidebar.number_input("Home Odds", value=2.5, step=0.1)
draw_odds = st.sidebar.number_input("Draw Odds", value=3.0, step=0.1)
away_odds = st.sidebar.number_input("Away Odds", value=3.5, step=0.1)

input_features = np.array([[home_odds, draw_odds, away_odds]])

# Predictions
st.subheader("Predictions")

# Match Result Prediction
if st.button("Predict Match Result"):
    match_result = model_match_result.predict(input_features)
    result_map = {0: "Draw", 1: "Away Win", 2: "Home Win"}
    st.write(f"Predicted Match Result: **{result_map[match_result[0]]}**")

# Over 2.5 Goals Prediction
if st.button("Predict Over 2.5 Goals"):
    over_2_5_prediction = model_o2_5.predict(input_features)
    st.write(f"Predicted Over 2.5 Goals: **{'Yes' if over_2_5_prediction[0] == 1 else 'No'}**")

# Optional: Display Features
if st.checkbox("Show Input Features"):
    st.write("Input features are:")
    st.write(input_features)
