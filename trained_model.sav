import streamlit as st
import pandas as pd
import pickle

# Load the pickled model
with open('trained_model.sav', 'rb') as f:
    loaded_model = pickle.load(f)

# Create a function to make predictions
def predict_chd(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    df = pd.DataFrame(data, columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    prediction = loaded_model.predict(df)
    return prediction

# Add a title and a short description
st.title('Heart Disease Prediction')
st.write('Enter the patient details below:')

# Add input fields for patient details
age = st.number_input('Age', min_value=1, max_value=120)
sex = st.radio('Sex', options=['Male', 'Female'])
cp = st.selectbox('Chest pain type', options=[0, 1, 2, 3])
trestbps = st.number_input('Resting blood pressure (mm Hg)', min_value=1, max_value=300)
chol = st.number_input('Serum cholesterol (mg/dl)', min_value=1, max_value=1000)
fbs = st.radio('Fasting blood sugar > 120 mg/dl', options=['Yes', 'No'])
restecg = st.selectbox('Resting electrocardiographic results', options=[0, 1, 2])
thalach = st.number_input('Maximum heart rate achieved', min_value=1, max_value=300)
exang = st.radio('Exercise induced angina', options=['Yes', 'No'])
oldpeak = st.number_input('ST depression induced by exercise relative to rest', min_value=0.0, max_value=10.0)
slope = st.selectbox('The slope of the peak exercise ST segment', options=[0, 1, 2])
ca = st.selectbox('Number of major vessels (0-3) colored by flourosopy', options=[0, 1, 2, 3])
thal = st.selectbox('Thalassemia', options=[0, 1, 2, 3])

# Call the predict_chd function and display the prediction
if st.button('Predict'):
    sex = 1 if sex == 'Male' else 0
    fbs = 1 if fbs == 'Yes' else 0
    exang = 1 if exang == 'Yes' else 0
    prediction = predict_chd(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    if prediction[0] == 1:
        st.write
