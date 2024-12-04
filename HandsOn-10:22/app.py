import streamlit as st
import numpy as np
import xgboost as xgb
import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load environment variables
load_dotenv()
mongo_uri = os.getenv("MONGO_DB_CONN_URL")

# Load pre-trained XGBoost models
female_structured_model = xgb.Booster()
female_structured_model.load_model('/Users/sreevardhanreddysoma/Desktop/HandsOn-10:22/DLOptimized_model.h5')
male_structured_model = xgb.Booster()
male_structured_model.load_model('/Users/sreevardhanreddysoma/Desktop/HandsOn-10:22/DLOptimized_model.h5')

# Connect to MongoDB
client = MongoClient(mongo_uri)
db = client['DiabetesRepo']
collection = db['Diabetes Prediction Data']

# Define the LLaMA model path (replace with your actual path)
llama_model_path = "/Users/sreevardhanreddysoma/.llama/checkpoints/Llama3.1-70B"

# Load LLaMA model and tokenizer
@st.cache_resource
def load_llama_model():
    tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
    model = AutoModelForCausalLM.from_pretrained(llama_model_path)
    return model, tokenizer

model, tokenizer = load_llama_model()

# Generate recommendations using the LLaMA model
def generate_llama_recommendations(predicted_class, input_data):
    class_info = {
        0: "no diabetes",
        1: "prediabetes",
        2: "type 2 diabetes",
        3: "gestational diabetes"
    }
    input_summary = ', '.join([f"{k}: {v}" for k, v in input_data.items()])
    prompt = (f"Given a patient profile: {input_summary} and predicted diagnosis: {class_info[predicted_class]}, "
              "provide personalized health recommendations based on this information.")

    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(inputs['input_ids'], max_length=150)
    recommendations = tokenizer.decode(output[0], skip_special_tokens=True)
    return recommendations

# Streamlit session state for gender selection
if 'gender' not in st.session_state:
    st.session_state.gender = None

# Helper function to style sections
def styled_header(title, subtitle=None):
    st.markdown(f"<h1 style='color: #4CAF50;'>{title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<h3 style='color: #555;'>{subtitle}</h3>", unsafe_allow_html=True)

# Custom label encoding function
def custom_label_encode(value, key):
    encoding_dicts = {
        'BPLevel': {"Normal": 0, "Low": 1, "High": 2},
        'PhysicallyActive': {"None": 0, "Less than half an hour": 1, "More than half an hour": 2, "One hour or more": 3},
        'HighBP': {"No": 0, "Yes": 1},
        'Gestation in previous Pregnancy': {"No": 0, "Yes": 1},
        'PCOS': {"No": 0, "Yes": 1},
        'Smoking': {"No": 0, "Yes": 1},
        'RegularMedicine': {"No": 0, "Yes": 1},
        'Stress': {"No": 0, "Yes": 1}
    }
    return encoding_dicts.get(key, {}).get(value, value)

# Class labels for predictions
class_labels = {
    0: "No diabetes",
    1: "Prediabetes",
    2: "Type 2 diabetes",
    3: "Gestational diabetes"
}

# Gender selection page
if st.session_state.gender is None or st.session_state.gender == "Select your gender":
    styled_header("Diabetes Prediction App")
    st.session_state.gender = st.selectbox("Select your gender", options=["Select your gender", "Male", "Female"])
    if st.session_state.gender != "Select your gender":
        st.rerun()

# Gender-specific questions
else:
    styled_header(f"Questionnaire for {st.session_state.gender} Patients")
    st.markdown("Please fill out the details carefully. Accurate information helps in better prediction.")
    
    if st.button("Back to Gender Selection", key="back"):
        st.session_state.gender = None
        st.rerun()

    gender_specific_data = {}
    
    # Number input helper function
    def number_input_with_none(label):
        user_input = st.text_input(label)
        return float(user_input) if user_input else None

    age = number_input_with_none("Enter your age")
    physically_active = st.selectbox("How much physical activity do you get daily?", options=["", "Less than half an hour", "None", "More than half an hour", "One hour or more"])
    bp_level = st.selectbox("What is your blood pressure level?", options=["", "High", "Normal", "Low"])
    high_bp = st.selectbox("Have you been diagnosed with high blood pressure?", options=["", "Yes", "No"])
    sleep = number_input_with_none("Average sleep time per day (in hours)")
    sound_sleep = number_input_with_none("Average hours of sound sleep")
    height_in = number_input_with_none("Height (in inches)")
    weight_lb = number_input_with_none("Weight (in pounds)")

    if height_in and weight_lb:
        bmi = (weight_lb * 703) / (height_in ** 2)
        st.success(f"Your calculated BMI is: **{bmi:.2f}**")
    else:
        st.warning("Please provide both height and weight for BMI calculation.")

    if st.session_state.gender == "Female":
        pregnancies = number_input_with_none("Number of pregnancies")
        gestation_history = st.selectbox("Have you had gestational diabetes?", options=["", "Yes", "No"])
        pcos = st.selectbox("Have you been diagnosed with PCOS?", options=["", "Yes", "No"])
        gender_specific_data = {'Pregnancies': pregnancies, 'Gestation in previous Pregnancy': gestation_history, 'PCOS': pcos}
        
    elif st.session_state.gender == "Male":
        smoking = st.selectbox("Do you smoke?", options=["", "Yes", "No"])
        regular_medicine = st.selectbox("Do you take regular medicine for diabetes?", options=["", "Yes", "No"])
        stress = st.selectbox("Do you experience high levels of stress?", options=["", "Yes", "No"])
        gender_specific_data = {'Smoking': smoking, 'RegularMedicine': regular_medicine, 'Stress': stress}

    input_data_dict = {
        'Age': age,
        'PhysicallyActive': physically_active,
        'BPLevel': bp_level,
        'HighBP': high_bp,
        'Sleep': sleep,
        'SoundSleep': sound_sleep,
        'BMI': bmi if height_in and weight_lb else None
    }
    input_data_dict.update(gender_specific_data)

    if st.button("Submit"):
        input_data_encoded = {}
        for key, value in input_data_dict.items():
            if isinstance(value, str) and value:
                input_data_encoded[key] = custom_label_encode(value, key)
            else:
                input_data_encoded[key] = value

        st.warning(f"Encoded categorical data: {input_data_encoded}")

        # Convert data for prediction
        input_data_df = pd.DataFrame([input_data_encoded])
        expected_feature_names = ['Age', 'HighBP', 'PhysicallyActive', 'BMI', 'Sleep', 'SoundSleep', 'BPLevel', 'Pregnancies', 'Gestation in previous Pregnancy', 'PCOS']
        input_data_df = input_data_df.reindex(columns=expected_feature_names)
        d_matrix = xgb.DMatrix(data=input_data_df)

        if st.session_state.gender == "Female":
            structured_probs = female_structured_model.predict(d_matrix)
        else:
            structured_probs = male_structured_model.predict(d_matrix)

        predicted_class = np.argmax(structured_probs)
        st.success(f"The predicted class is: {class_labels[predicted_class]} with probability {np.max(structured_probs):.2f}")

        # Use LLaMA for dynamic recommendations
        llama_recommendations = generate_llama_recommendations(predicted_class, input_data_dict)
        st.info("### Recommendations")
        st.write(llama_recommendations)

        # Insert data to MongoDB
        entry = {
            **input_data_encoded,
            'Gender': st.session_state.gender,
            'class_probabilities': structured_probs.tolist(),
            'prediction': int(predicted_class),
            'diagnosis': class_labels[predicted_class]
        }
        collection.insert_one(entry)
        st.success("Data successfully uploaded to MongoDB!")