import requests

# Flask API endpoint
API_URL = "http://127.0.0.1:5000/predict"

# Example input (make sure all features are included)
sample_data = {
    "fever": 1,
    "vomiting": 1,
    "weakness": 1,
    "stomach_pain": 0,
    "diarrhea": 1,
    "headache": 0,
    "jaundice": 0
}

headers = {'Content-Type': 'application/json'}

print(f"Sending data to {API_URL}...")
print(f"Input: {sample_data}")

try:
    response = requests.post(API_URL, json=sample_data, headers=headers)

    if response.status_code == 200:
        result = response.json()
        print("\n--- API Prediction Success ---")

        # Show top predictions
        for pred in result.get('top_predictions', []):
            print(f"Disease: {pred['disease']}, Probability: {pred['probability']:.2f}")

        # Show all probabilities (for debugging)
        print("\n--- All Class Probabilities ---")
        for disease, prob in result.get('all_probabilities', {}).items():
            print(f"{disease}: {prob:.3f}")

    else:
        print("\n--- API Error ---")
        print(f"Status Code: {response.status_code}")
        print(f"Error Response: {response.text}")

except requests.exceptions.ConnectionError:
    print("\n❌ Error: Could not connect to the API server.")
    print("Make sure api_server.py is running in another terminal.")
