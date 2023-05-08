pip install joblib
import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.joblib')

# Define the web interface
def main():
    # Add a title to the app
    st.title('Heart Disease Prediction')

    # Add a form for users to enter input data
    st.subheader('Input Parameters')
    age = st.number_input('Age')
    sex = st.selectbox('Sex', ['Male', 'Female'])
    cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
    trestbps = st.number_input('Resting Blood Pressure')
    chol = st.number_input('Cholesterol')
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['True', 'False'])
    restecg = st.selectbox('Resting ECG', ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy'])
    thalach = st.number_input('Maximum Heart Rate Achieved')
    exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
    oldpeak = st.number_input('ST Depression Induced by Exercise')

    # Convert the input data into a numpy array
    data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]])

    # Add a button to run the model and make a prediction
    if st.button('Predict'):
        # Use the model to make a prediction
        prediction = model.predict(data)[0]
        # Show the prediction result
        st.subheader('Prediction Result')
        st.write('The patient is likely to have heart disease.' if prediction else 'The patient is not likely to have heart disease.')

if __name__ == '__main__':
    main()
