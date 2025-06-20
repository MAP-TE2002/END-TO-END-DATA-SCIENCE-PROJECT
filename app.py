# %%
pip install --upgrade scikit-learn

# %%
# retrain_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv("heart.csv")

# Split
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Save with your environment's format
joblib.dump(model, "heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")


# %%
from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    features = np.array([[
        data["age"], data["sex"], data["cp"], data["trestbps"], data["chol"],
        data["fbs"], data["restecg"], data["thalach"], data["exang"],
        data["oldpeak"], data["slope"], data["ca"], data["thal"]
    ]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    return jsonify({"prediction": int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)

