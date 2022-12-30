# -*- coding: utf-8 -*-

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
#from pathlib import path
import streamlit_authenticator as stauth
#user authentication
import pandas as pd
import random

# loading the saved models

diabetes_model = pickle.load(open('/Users/tarunkumar/Desktop/Multiple Disease Prediction System/saved models/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open('/Users/tarunkumar/Desktop/Multiple Disease Prediction System/saved models/heart_disease_model.sav','rb'))

parkinsons_model = pickle.load(open('/Users/tarunkumar/Desktop/Multiple Disease Prediction System/saved models/parkinsons_model.sav', 'rb'))

breast_model=pickle.load(open('/Users/tarunkumar/Desktop/Multiple Disease Prediction System/saved models/breast_cancer_model.sav','rb'))
# sidebar for navigation
breastcancer_doc=pd.read_csv("/Users/tarunkumar/Desktop/Multiple Disease Prediction System/brdoc.csv")
breastcancer_doc=breastcancer_doc.drop("index",axis='columns')
diab_doc=pd.read_csv("/Users/tarunkumar/Desktop/Multiple Disease Prediction System/diabdoc.csv")
diab_doc=diab_doc.drop("index",axis="columns")
heart_doc=pd.read_csv("/Users/tarunkumar/Desktop/Multiple Disease Prediction System/heartdoc.csv")
heart_doc = heart_doc.drop("index", axis='columns')
with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System',
                          
                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction',
                           'Breast Cancer Prediction'],
                          icons=['activity','heart','person','person'],
                          default_index=0)
if 'random' not in st.session_state:
    st.session_state['random'] = random.randint(0,130)
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction using ML')
   
    st.image('/Users/tarunkumar/Desktop/Multiple Disease Prediction System/Diabetes_HeartDisease_shutterstock_313336877.jpg')
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies (0-17)')
        
    with col2:
        Glucose = st.text_input('Glucose Level(0-199)')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value(0-122)')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value(0-110)')
    
    with col2:
        Insulin = st.text_input('Insulin Level(0-744)')
    
    with col3:
        BMI = st.text_input('BMI value(0-81)')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value(0-078-2.43)')
    
    with col2:
        Age = st.text_input('Age of the Person(21-81)')
    
    
    # code for Prediction
    diab_diagnosis = ''
    doc=""
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
          doc=diab_doc.sample(n=5,random_state=st.session_state['random'])   
        else:
          diab_diagnosis = 'The person is not diabetic'  
          doc="no doctor needed"
    st.success(diab_diagnosis)
    st.write(doc) 
    
    




# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    st.image('/Users/tarunkumar/Desktop/Multiple Disease Prediction System/Heart-Disease.jpg')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex (1-male,0-female)')
        
    with col3:
        cp = st.text_input('Chest Pain types (1-typical angina,2-atypical angina,3-non-aginal pain,4-asympotic)')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure (mmHg(unit))')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl then 1 else 0')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results (0-normal,1-having ST-T abnormality,2-left ventricular hyperthorophy')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina (1-yes,0-no')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment (1-upsloping,2-flat,3-downsloping)')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy(0-3)')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    doc1=""
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
          doc1=heart_doc.sample(n=5,random_state=st.session_state['random']) 
        else:
          heart_diagnosis = 'The person does not have any heart disease'
          doc1="no doctor needed"
        
    st.success(heart_diagnosis)
    st.write(doc1)
        
    
    

# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    st.image('/Users/tarunkumar/Desktop/Multiple Disease Prediction System/parkinsons-disease-cartoon-parkinsons-disease-cartoon-icon-vector-illustration-graphic-design-108757311 (1).jpg')
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz) (88-260)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz) (102-592)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz) (65-240)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)(0.0016-0.0331)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)(7e-06-.00026)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP (0.0068-0.2144)')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ  (0.00092-0.1958)')
        
    with col3:
        DDP = st.text_input('Jitter:DDP (0.00204-0.06433)')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer  (0.00954-0.11908)')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB) (0.085-1.302)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3 (0.00455-0.05647) ')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5 (0.0057-0.0794)')
        
    with col3:
        APQ = st.text_input('MDVP:APQ (0.00719-0.13778)')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA (0.01364-0.16942)')
        
    with col5:
        NHR = st.text_input('NHR (0.00065-0.31482)')
        
    with col1:
        HNR = st.text_input('HNR (8.441-33.047)')
        
    with col2:
        RPDE = st.text_input('RPDE (0.25657-0.685151)')
        
    with col3:
        DFA = st.text_input('DFA (0.574-0.825)')
        
    with col4:
        spread1 = st.text_input('spread1 (-7 (to) -2)')
        
    with col5:
        spread2 = st.text_input('spread2 (0.006247-0.450493)')
        
    with col1:
        D2 = st.text_input('D2 (1.423-3.67115)')
        
    with col2:
        PPE = st.text_input('PPE (.044539-0.5273)')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)


if (selected == 'Breast Cancer Prediction'):

    # page title
    st.title('Breast Cancer  Prediction using ML')
   
    st.image('/Users/tarunkumar/Desktop/Multiple Disease Prediction System/breast_cancer.jpg')
    # getting the input data from the user
    col1, col2 = st.columns(2)
    
    with col1:
        mean_radius = st.text_input('Enter the mean Radius (6.5-28.5)')
        
    with col2:
        mean_texture = st.text_input('Enter the mean Texture (9.5-39.5)')
    
    
    with col1:
        mean_perimeter = st.text_input('Enter the mean Perimeter(43-189)')
    
    with col2:
        mean_area = st.text_input('Enter the mean Area( 143-2501)')
    
    with col1:
        mean_smoothness = st.text_input('Enter the mean Smoothness (0.05-0.170 )')
    
    # code for Prediction
    breast_diagnosis = ''
    doc3=""
    # creating a button for Prediction
    
    if st.button('Breast Cancer Test Result'):
        breast_prediction = breast_model.predict([[mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness]])
        
        if (breast_prediction[0] == 1):
          breast_diagnosis = 'The person contain Breast Cancer'
          doc3=breastcancer_doc.sample(n=5,random_state=st.session_state['random']) 
        else:
          breast_diagnosis = 'The person not contain Breast Cancer'
          doc3="no doctor needed" 
    st.success(breast_diagnosis)
    st.write(doc3)

