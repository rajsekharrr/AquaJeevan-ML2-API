import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import operator

# --- 1. DATA GENERATION FUNCTION (MULTI-CLASS DISEASE) ---
def generate_symptom_data(num_rows=2000):
    """
    Generates a synthetic dataset of community health symptom reports.
    """
    print("Generating a synthetic dataset for the Symptom Model...")

    villages = [f'V{i:03d}' for i in range(1, 21)]
    data = {
        'patient_id': [f'P{i:04d}' for i in range(1, num_rows + 1)],
        'age': np.random.randint(1, 91, size=num_rows),
        'gender': np.random.choice(['Male', 'Female', 'Other'], size=num_rows, p=[0.45, 0.45, 0.1]),
        'location': np.random.choice(villages, size=num_rows),
        'fever': np.random.choice([0, 1], size=num_rows, p=[0.7, 0.3]),
        'vomiting': np.random.choice([0, 1], size=num_rows, p=[0.75, 0.25]),
        'weakness': np.random.choice([0, 1], size=num_rows, p=[0.6, 0.4]),
        'stomach_pain': np.random.choice([0, 1], size=num_rows, p=[0.7, 0.3]),
        'diarrhea': np.random.choice([0, 1], size=num_rows, p=[0.65, 0.35]),
        'headache': np.random.choice([0, 1], size=num_rows, p=[0.5, 0.5]),
        'jaundice': np.random.choice([0, 1], size=num_rows, p=[0.95, 0.05])
    }
    df = pd.DataFrame(data)

    # --- MULTI-CLASS LABELING RULE ---
    def assign_disease(row):
        if row['jaundice'] == 1 and row['fever'] == 1:
            return 'Hepatitis A'
        elif row['diarrhea'] == 1 and row['vomiting'] == 1 and row['weakness'] == 1:
            return 'Cholera'
        elif row['fever'] == 1 and row['stomach_pain'] == 1 and row['headache'] == 1:
            return 'Typhoid'
        else:
            return 'None'

    df['disease'] = df.apply(assign_disease, axis=1)

    # Add some noise to make dataset less biased
    noise_indices = np.random.choice(df.index, size=int(num_rows * 0.05), replace=False)
    possible_diseases = ['Hepatitis A', 'Cholera', 'Typhoid', 'None']
    for idx in noise_indices:
        df.at[idx, 'disease'] = np.random.choice(possible_diseases)

    return df


# --- 2. TRAINING THE ML MODEL AND SAVING ---
if __name__ == '__main__':
    print("Training the ML Model and saving the joblib file...")
    df = generate_symptom_data(2000)

    # Define symptom features and the target label
    symptom_features = ['fever', 'vomiting', 'weakness', 'stomach_pain', 'diarrhea', 'headache', 'jaundice']
    X = df[symptom_features]
    y = df['disease']

    # Check balance of target classes
    print("\n--- Disease Class Distribution ---")
    print(y.value_counts())

    # Stratified split keeps distribution balanced
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Use class_weight='balanced' to handle imbalance
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    # Save the trained model to a file
    model_filename = 'disease_model.joblib'
    joblib.dump(model, model_filename)
    print(f"\n✅ Successfully trained the model and saved it to '{model_filename}'")

    # Save feature names and classes for the API
    metadata = {
        'features': symptom_features,
        'classes': model.classes_.tolist()
    }
    joblib.dump(metadata, 'model_metadata.joblib')
    print("✅ Saved model metadata (features and classes).")

    # --- Model Evaluation ---
    y_pred = model.predict(X_test)
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=model.classes_))

    # --- 3. FUNCTION TO INPUT SYMPTOMS AND PREDICT TOP 2 DISEASES ---
    def input_and_predict():
        """
        Takes symptom inputs from the user and predicts the top 2 most likely diseases.
        """
        print("\n--- Running Interactive Prediction (Type 1 or 0) ---")
        new_patient = {}
        for symptom in symptom_features:
            while True:
                try:
                    value = input(f"{symptom.replace('_',' ').capitalize()}: ").strip()
                    if value == "":
                        v = np.random.choice([0, 1], p=[0.7, 0.3])  # Random if blank
                        print(f"  > No input given. Using random value: {v}")
                    else:
                        v = int(value)
                        if v not in [0, 1]:
                            raise ValueError
                    new_patient[symptom] = v
                    break
                except ValueError:
                    print("Invalid input! Enter 1 for Yes, 0 for No, or leave blank.")

        new_df = pd.DataFrame([new_patient])

        # Predict probabilities
        probabilities = model.predict_proba(new_df)[0]

        pred_dict = dict(zip(model.classes_, probabilities))
        sorted_preds = sorted(pred_dict.items(), key=operator.itemgetter(1), reverse=True)

        print("\n--- Top Predicted Diseases ---")
        top_count = 0
        for disease, prob in sorted_preds:
            if disease == 'None' and top_count > 0:
                continue
            if prob > 0.1:
                print(f"Predicted Disease: {disease} (Probability: {prob:.2f})")
                top_count += 1
            if top_count >= 2:
                break

        if top_count == 0:
            print("No specific waterborne disease was predicted with high confidence.")

    # --- 4. RUN THE PREDICTION ---
    input_and_predict()
