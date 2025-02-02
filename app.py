import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

df = pickle.load(open('df.pkl','rb'))
pipe = pickle.load(open('pipe.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('Stroke Prediction')

gender=st.selectbox('Gender',options=df['gender'].unique())
age=st.number_input('Age',value=None)
hypertension=st.selectbox('Hyper Tension',options=['Yes','No'])
heart_disease=st.selectbox('Heart Disease',options=['Yes','No'])
ever_married=st.selectbox('Married',options=['Yes','No'])
work_type=st.selectbox('Work Type',options=df['work_type'].unique())
Residence_type=st.selectbox('Residence Type',options=df['Residence_type'].unique())
avg_glucose_level=st.selectbox('Average Glucose Level',options=df['avg_glucose_level'].unique())
bmi=st.selectbox('BMI',options=df['bmi'].unique())
smoking_status=st.selectbox('Smoking Status',options=df['smoking_status'].unique())

dff = pd.DataFrame(
    {
        'gender':[gender],
        'age':[age],
        'hypertension':[hypertension],
        'heart_disease':[heart_disease],
        'ever_married':[ever_married],
        'work_type':[work_type],
        'Residence_type':[Residence_type],
        'avg_glucose_level':[avg_glucose_level],
        'bmi':[bmi],
        'smoking_status':[smoking_status]
    }
)
trans_data = pipe.transform(dff)

btn = st.button('Predict')

if btn:
    st.write("Possibility of Stroke " )
    if model.predict(trans_data)==1:
        st.title('Yes')
    else:
        st.title('No')        

         