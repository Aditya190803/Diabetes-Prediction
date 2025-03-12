import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import pickle
import os

# Load dataset (Pima Indians Diabetes Dataset)
DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMN_NAMES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

NORMAL_RANGES = {
    "Pregnancies": "0-15",
    "Glucose": "70-140 mg/dL",
    "BloodPressure": "80-120 mmHg",
    "SkinThickness": "10-50 mm",
    "Insulin": "15-276 μU/mL",
    "BMI": "18.5-24.9",
    "DiabetesPedigreeFunction": "0.1-2.5",
    "Age": "20-80 years"
}

@st.cache_resource
def load_data():
    data = pd.read_csv(DATA_URL, names=COLUMN_NAMES)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]
    return X, y

X, y = load_data()

# Split data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

@st.cache_resource
def load_model_file():
    MODEL_PATH = "diabetes_ann_model.h5"
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    else:
        # Build ANN model
        model = Sequential([
            Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        # Compile model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train model
        model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test))

        # Save model
        model.save(MODEL_PATH)
        return model

model = load_model_file()

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("Diabetes Prediction using ANN")
st.write("Enter the following details to check the likelihood of diabetes.")

# Display normal ranges in sidebar
st.sidebar.header("Normal Ranges")
for feature, value in NORMAL_RANGES.items():
    st.sidebar.write(f"**{feature}:** {value}")

# Initialize session state for inputs
if "inputs" not in st.session_state:
    st.session_state.inputs = {
        "Pregnancies": "0",
        "Glucose": "100",
        "BloodPressure": "80",
        "SkinThickness": "20",
        "Insulin": "30",
        "BMI": "22",
        "DiabetesPedigreeFunction": "0.5",
        "Age": "30"
    }

# Improved Input fields with text boxes
data_input = []
for feature in COLUMN_NAMES[:-1]:  # Exclude Outcome
    st.session_state.inputs[feature] = st.text_input(
        f"{feature}", st.session_state.inputs[feature]
    )
    data_input.append(st.session_state.inputs[feature])

# Predict button
if st.button("Predict"):
    try:
        input_data = np.array([float(i) for i in data_input]).reshape(1, -1)
        input_data = scaler.transform(input_data)  # Normalize
        prediction = model.predict(input_data)[0][0]
        result = "Diabetic" if prediction > 0.5 else "Non-Diabetic"
        st.success(f"Prediction: {result}")
    except ValueError:
        st.error("Please enter valid numerical values.")
