import streamlit as st
import joblib
import pandas as pd

# Load the models
model_o2_5 = joblib.load('model_o2_5.pkl')
model_result = joblib.load('model_result.pkl')

st.title("FootyPred: Football Match Predictor")

# Input fields
home_team = st.text_input("Home Team")
away_team = st.text_input("Away Team")
date = st.date_input("Match Date")
home_odds = st.number_input("Home Odds", min_value=0.0)
draw_odds = st.number_input("Draw Odds", min_value=0.0)
away_odds = st.number_input("Away Odds", min_value=0.0)

if st.button("Predict"):
    # Create a DataFrame for the input features
    features = pd.DataFrame([{
        'HomeOdds': home_odds,
        'DrawOdds': draw_odds,
        'AwayOdds': away_odds
    }])

    # Make predictions
    prediction_o2_5 = model_o2_5.predict(features)
    prediction_result = model_result.predict(features)

    # Display predictions
    st.write(f"Prediction for Over 2.5 Goals: {'Yes' if prediction_o2_5[0] == 1 else 'No'}")
    st.write(f"Predicted Match Result: {prediction_result[0]}")
    st.write(f"Match between {home_team} and {away_team} on {date}")
