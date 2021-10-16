#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st 
from pickle import dump
from pickle import load
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.datasets import make_classification
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
da= pd.read_csv('C:/Users/User/Downloads/jais/shipments.csv')
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
dump(model,open('model_ran.fit.sav', 'wb'))

loaded_model=load(open('model_ran.fit.sav' ,'rb'))
result = loaded_model.score(X,y)
print(result)
# In[ ]:




