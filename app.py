import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pickle



model = tf.keras.models.load_model('regression.h5')

with open('Le_regression.pkl','rb') as f:
    gender_le = pickle.load(f)
with open('ohe_regression.pkl','rb') as f:
    geo_ohe = pickle.load(f)
with open('ss_regression.pkl','rb') as f:
    ss = pickle.load(f)

st.title('Customer Salary Prediction')

geography = st.selectbox('Geography',geo_ohe.categories_[0])
gender = st.selectbox('Gender',gender_le.classes_)
age = st.slider('Age',18,100)
balance = st.number_input('Balance',0)
credit_score = st.number_input('Credit Score',0)
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',0,4)
has_cr_card = st.selectbox('Has Car Card',[0,1])
is_active_member = st.selectbox('Is Active member',[0,1])
excited = st.selectbox('Exited',[0,1])


input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[gender_le.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'Exited':[excited]
}
)

ohe_geo = geo_ohe.transform([[geography]]).toarray()
ohe_geo_col = geo_ohe.get_feature_names_out(['Geography'])

geo_df = pd.DataFrame(ohe_geo,columns = ohe_geo_col)

input_data = pd.concat([input_data.reset_index(drop=True),geo_df],axis=1)

input_data_scaled = ss.transform(input_data)

prediction = model.predict(input_data_scaled)
predicted_salary = prediction[0][0]

st.write(f'Predicted Estimated Salary: ${predicted_salary:.2f}')














