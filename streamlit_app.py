import streamlit as st
from joblib import dump, load


st.title("Deploying the model")
LABELS = ['setosa', 'versicolor', 'virginica']

classifier = load("IrisClassifier.joblib")

#%%

sp_l = st.slider('sepal length (cm)',min_value=0, max_value=10)
sp_w = st.slider('sepal width (cm)',min_value=0, max_value=10)
pt_l = st.slider('petal length (cm)',min_value=0, max_value=10)
pt_w = st.slider('petal length (cm)',min_value=0, max_value=10)

prediction = classifier.predict(sp_l,sp_w,pt_l,pt_w)

st.write(LABELS[prediction[0]])