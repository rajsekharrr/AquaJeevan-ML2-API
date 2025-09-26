import joblib, pandas as pd

MODEL = joblib.load("disease_model.joblib")
META = joblib.load("model_metadata.joblib")

features = META["features"]
classes = META["classes"]

sample = pd.DataFrame([[1,1,0,0,1,0,0]], columns=features)

probs = MODEL.predict_proba(sample)[0]
print("Classes:", classes)
print("Probabilities:", probs)
