import streamlit as st
import numpy as np
from joblib import dump, load
import keras
import tensorflow

st.write(f"Keras version {keras.__version__}")
st.write(f"Keras version {tensorflow.__version__}")


st.title("Deploying the model")
LABELS = ['setosa', 'versicolor', 'virginica']

DecisionTreeClassifier = load("IrisClassifier.joblib")
NeuralNetworkClassifier = load("NeuralNetwork.joblib")

#%%

sp_l = st.slider('sepal length (cm)',min_value=0, max_value=10)
sp_w = st.slider('sepal width (cm)',min_value=0, max_value=10)
pt_l = st.slider('petal length (cm)',min_value=0, max_value=10)
pt_w = st.slider('petal width (cm)',min_value=0, max_value=10)

DT_Predict = DecisionTreeClassifier.predict([[sp_l,sp_w,pt_l,pt_w]])

NN_Predict = NeuralNetworkClassifier.predict([[sp_l,sp_w,pt_l,pt_w]])

st.write(f"Decision Tree predicts {LABELS[DT_Predict[0]]}")
st.write(f"Neural Network predicts {LABELS[np.argmax(NN_Predict)]}")