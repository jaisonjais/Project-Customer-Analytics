# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:01:17 2020

"""

import pandas as pd
import numpy as np
import streamlit as st 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

st.title('Customer Analytics Using Random Forest Algorithm')

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
st.write(dfp)
dfp["Warehouse_block"].replace({"A":0,"B":1,"C":2,"D":3,"F":4},inplace=True)
dfp["Mode_of_Shipment"].replace({"Flight":0,"Road":1,"Ship":2},inplace=True)
dfp["Product_importance"].replace({"high":0,"low":1,"medium":2},inplace=True)
dfp["Gender"].replace({"M":1,"F":0},inplace=True)

da = pd.read_csv("shipments.csv")


da.drop(["ID"],inplace=True,axis = 1)
da= da.dropna()  
#label encoding
da["Warehouse_block"].replace({"A":0,"B":1,"C":2,"D":3,"F":4},inplace=True)
da["Mode_of_Shipment"].replace({"Flight":0,"Road":1,"Ship":2},inplace=True)
da["Product_importance"].replace({"high":0,"low":1,"medium":2},inplace=True)
da["Gender"].replace({"M":1,"F":0},inplace=True)
da.Gender=da.Gender.astype(str).str.strip()
da = da.rename({'Reached.on.Time_Y.N':'target'},axis = 1)

# Class count
count_class_0, count_class_1 = da.target.value_counts()

# Divide by class
df_class_0 = da[da['target'] == 0]
df_class_1 = da[da['target'] == 1]


df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)
norm_data=df_test_over[['Cost_of_the_Product','Discount_offered','Weight_in_gms']]

ntdata= (norm_data - np.min(norm_data)) / (np.max(norm_data) - np.min(norm_data))
dat=df_test_over.drop(['Cost_of_the_Product','Discount_offered','Weight_in_gms'],axis=1)
newdata=pd.concat([ntdata,dat],axis=1)


y = newdata.target
x = newdata.drop('target', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.04,shuffle=True,random_state=45)


from imblearn.over_sampling import SMOTE

# create the  object with the desired sampling strategy.
smote = SMOTE(sampling_strategy='not minority')
x_train_enn, y_train_enn=smote.fit_resample(x_train,y_train)
model_ran = RandomForestClassifier(n_estimators=10)




model_ran.fit(x_train_enn, y_train_enn)


prediction = model_ran.predict(dfp)
prediction_proba = model_ran.predict_proba(dfp)


df1=pd.DataFrame(prediction,columns=['0'])
df1.loc[df1['0'] == 1, 'Product Will Reach On Time'] = 'No'
df1.loc[df1['0'] == 0, 'Product Will Reach On Time'] = 'Yes'
st.write(df1)
