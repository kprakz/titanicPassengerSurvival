import streamlit as st
import joblib
import numpy as np

#load the model and scaler
log_reg = joblib.load('models/log_reg_model.pkl')
scaler = joblib.load('models/scaler.pkl')
tree_model = joblib.load('models/tree_model.pkl')

# Title
st.title('Titanic Survival Prediction')

#Input fields
age = st.number_input('Age', min_value=1, max_value=100, value=25)
fare = st.number_input('Fare', min_value=1, max_value=600, value=50)
Pclass = st.selectbox('Pclass', [1, 2, 3])
sex = st.selectbox('Sex', ["Male","Female"])
embarked = st.selectbox('Embarked', ["C","Q","S"])
family_size = st.slider('Family Size', min_value=0, max_value=10, value=1)
deck = st.selectbox('Deck', ["A","B","C","D","E","F","G","M","T", "Unknown"])
title = st.selectbox('Title', ["Mr","Miss","Mrs","Master","Rare"])

#Encoding categorical variables
title_map = {     "Don": [1, 0, 0, 0, 0, 0, 0, 0],
    "Jonkheer": [0, 1, 0, 0, 0, 0, 0, 0],
    "Officer": [0, 0, 1, 0, 0, 0, 0, 0],
    "Master": [0, 0, 0, 1, 0, 0, 0, 0],
    "Sir": [0, 0, 0, 0, 1, 0, 0, 0],
    "Mr": [0, 0, 0, 0, 0, 1, 0, 0],
    "Miss": [0, 0, 0, 0, 0, 0, 1, 0],
    "Mrs": [0, 0, 0, 0, 0, 0, 0, 1]
      }

title_encoding = title_map[title]

#Normalise age
AGE_MEAN = 29.699118
AGE_STD = 14.526497
NORMALISED_AGE = (age - AGE_MEAN) / AGE_STD

#convert input to model input format
sex = 1 if sex == "Female" else 0

#Log transform fare
LOG_FARE = np.log(fare + 1)

#Pclass encoding 
Pclass_2 = 1 if Pclass == 2 else 0
Pclass_3 = 1 if Pclass == 3 else 0

#Deck encoding
deck_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T':8, 'Unknown': 0}
deck_value = deck_mapping[deck]

#IsAlone
is_alone = 1 if family_size == 0 else 0

#Embarked encoding
embarked_Q = 1 if embarked == "Q" else 0 # no need to mention C as its implied when both Q and S are 0
embarked_S = 1 if embarked == "S" else 0

#Apply scaling
scaled_faetures = scaler.transform(np.array([[NORMALISED_AGE, LOG_FARE]])).flatten()

# Create the final feature array
features = np.hstack([
    0,
    title_encoding,
    [scaled_faetures[0],sex, scaled_faetures[1], Pclass_2, Pclass_3],
    deck_value,
    [family_size, is_alone, embarked_Q, embarked_S]
    ])

#Predict button
if st.button("Predict"):
    prediction = log_reg.predict(features.reshape(1, -1))
    st.write("The predicted survival probability is: ", prediction[0])
    st.write("The predicted survival probability using tree model is: ", tree_model.predict(features.reshape(1, -1))[0])