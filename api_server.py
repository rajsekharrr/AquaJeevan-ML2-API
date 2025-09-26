import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# --- 1. Load the Model and Metadata ---
try:
    MODEL = joblib.load('disease_model.joblib')
    METADATA = joblib.load('model_metadata.joblib')

    FEATURE_NAMES = METADATA['features']
    DISEASE_CLASSES = METADATA['classes']

    print("✅ Model and metadata loaded successfully.")
    print(f"Expected features: {FEATURE_NAMES}")
    print(f"Possible diseases: {DISEASE_CLASSES}")

except FileNotFoundError:
    print("❌ Error: disease_model.joblib or model_metadata.joblib not found.")
    print("Please ensure you have run 'aquajeevan_ml2.py' first.")
    exit()

# --- 2. API Endpoint for Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts a POST request with symptom data and returns disease probabilities.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # --- Data Validation ---
        missing_features = [f for f in FEATURE_NAMES if f not in data]
        if missing_features:
            return jsonify({
                'error': 'Missing required features',
                'missing': missing_features,
                'required': FEATURE_NAMES
            }), 400

        # --- Prepare Input ---
        input_data = [data.get(feature) for feature in FEATURE_NAMES]
        input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)

        # --- Prediction ---
        probabilities = MODEL.predict_proba(input_df)[0]

        # Map probabilities to disease names
        pred_dict = dict(zip(DISEASE_CLASSES, probabilities))

        # Sort by probability
        sorted_preds = sorted(pred_dict.items(), key=lambda item: item[1], reverse=True)

        # Top 2 predictions (ignoring "None" unless it's the only choice)
        top_predictions = []
        top_count = 0
        for disease, prob in sorted_preds:
            if disease == 'None' and top_count > 0:
                continue
            if prob >= 0.1:
                top_predictions.append({
                    "disease": disease,
                    "probability": round(prob, 3)
                })
                top_count += 1
            if top_count >= 2:
                break

        # --- Response ---
        return jsonify({
            'status': 'success',
            'top_predictions': top_predictions,
            'all_probabilities': pred_dict  # 👈 useful for debugging
        }), 200

    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

# --- 3. Run the Server ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
