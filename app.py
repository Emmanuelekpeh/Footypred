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
    features = pd.DataFrame([data['features']])
    prediction_o2_5 = model_o2_5.predict(features)
    prediction_result = model_result.predict(features)
    
    # Include the original input data (team names and date) in the response
    response = {
        'prediction_o2_5': prediction_o2_5[0],
        'prediction_result': prediction_result[0],
        'home_team': data['features']['HomeTeam'],
        'away_team': data['features']['AwayTeam'],
        'date': data['features']['Date']
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
