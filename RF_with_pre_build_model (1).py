# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:01:17 2020

"""



import pandas as pd
import streamlit as st 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from pickle import dump
from pickle import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
st.title('Customer Analytics Using: Random forest')

st.sidebar.header('User Input Parameters')

def user_input_features():
    Customer_care_calls = st.sidebar.selectbox('Customer_care_calls',('2','3','4','5','6','7'))
    Customer_rating= st.sidebar.selectbox('Customer_rating',('1','2','3','4','5'))
    Prior_purchases = st.sidebar.selectbox('Prior_purchases',('2','3','4','5','6','7','8','10'))
    Gender = st.sidebar.selectbox('Gender',('F','M'))
    Warehouse_block = st.sidebar.selectbox('Warehouse_block',('A','B','C','D','F'))
    Mode_of_Shipment = st.sidebar.selectbox('Mode_of_Shipment',('Flight','Road','Ship'))
    Product_importance = st.sidebar.selectbox('Product_importance',('high','medium','low'))
    Cost_of_the_Product = st.sidebar.number_input("Cost_of_the_Product")
    Discount_offered = st.sidebar.number_input("Discount_offered")
    Weight_in_gms = st.sidebar.number_input("Weight_in_gms")
    data = {'Customer_care_calls':Customer_care_calls,
            'Customer_rating':Customer_rating,
            'Prior_purchases':Prior_purchases,
            'Gender':Gender,
            'Warehouse_block':Warehouse_block,
            'Mode_of_Shipment':Mode_of_Shipment,
            'Product_importance':Product_importance,
            'Cost_of_the_Product ':Cost_of_the_Product ,
            'Discount_offered ':Discount_offered ,
            'Weight_in_gms':Weight_in_gms}
    features = pd.DataFrame(data,index = [0])
    return features 
    
dfp = user_input_features()
st.subheader('User Input parameters')
st.write(df)
dfp["Warehouse_block"].replace({"A":0,"B":1,"C":2,"D":3,"F":4},inplace=True)
dfp["Mode_of_Shipment"].replace({"Flight":0,"Road":1,"Ship":2},inplace=True)
dfp["Product_importance"].replace({"high":0,"low":1,"medium":2},inplace=True)
dfp["Gender"].replace({"M":1,"F":0},inplace=True)

df.Gender=df.Gender.astype(str).str.strip()


# load the model from disk
loaded_model = load(open('model_ran.sav', 'rb'))

prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)



df1=pd.DataFrame(prediction,columns=['0'])
df1.loc[df1['0'] == 1, 'Shipment Will Reach On Time'] = 'No'
df1.loc[df1['0'] == 0, 'Shipment Will Reach On Time'] = 'Yes'
st.write(df1)



