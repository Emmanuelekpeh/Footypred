from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the models
model_o2_5 = joblib.load('model_o2_5.pkl')
model_result = joblib.load('model_result.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Extract features from the request
    features = pd.DataFrame([{
        'HomeOdds': data['HomeOdds'],
        'DrawOdds': data['DrawOdds'],
        'AwayOdds': data['AwayOdds']
    }])
    
    prediction_o2_5 = model_o2_5.predict(features)
    prediction_result = model_result.predict(features)
    
    # Include the original input data in the response
    response = {
        'prediction_o2_5': prediction_o2_5[0],
        'prediction_result': prediction_result[0],
        'home_team': data['HomeTeam'],
        'away_team': data['AwayTeam'],
        'date': data['Date']
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
