# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 19:39:22 2019

@author: go home
"""

import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from sklearn import datasets, svm, cross_validation, tree, preprocessing, metrics
import sklearn.ensemble as ske
import tensorflow as tf
from tensorflow.contrib import skflow
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import time

titan_Df = pd.read_excel(r'C:\titanic3.xls', 'titanic3', index_col=None, na_values=['NA'])
titan_Df.head() #Print top 5 lines
mean_overall_servi=titan_Df['survived'].mean() #Only 38% of passengers survived
Mean_byClass_BySex = titan_Df.groupby(['pclass','sex']).mean() #Shows survived passengers by pclass and by sex
Mean_byClass_BySex['survived'].plot.bar() #plotting the results


titan_Df.count() #ny doing this, we'll notice thet there are some missing values.we should clean our data in order to get our data balanced
titan_Df = titan_Df.drop(['body','cabin','boat'], axis=1)
titan_Df["home.dest"] = titan_Df["home.dest"].fillna("NA") #replace 'NAN' by 'NA'
titan_Df = titan_Df.dropna() 
titan_Df.count()

processed_df = titan_Df.copy()
Mo= preprocessing.LabelEncoder() # here we'll transform the columns sex and embarked, which are string datatype,into integer . 
processed_df.sex = Mo.fit_transform(processed_df.sex)  
processed_df.embarked = Mo.fit_transform(processed_df.embarked)
processed_df = processed_df.drop(['name','ticket','home.dest'],axis=1) #The “name”, “ticket”, and “home.dest” columns consist of non-categorical string values
###################Model#########################
X = processed_df.drop(['survived'], axis=1).values
y = processed_df['survived'].values #y contain only the 'survived' col processed
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.33) #train  our data
model = tree.DecisionTreeClassifier(max_depth=10) 
model.fit (X_train, y_train) 
ypredict=model.predict(X_test)
acc=accuracy_score(y_test,ypredict)
#############►RockCurve##########################
fpr,tpr,T=metrics.roc_curve(y_test,ypredict,pos_label=1)
lin_auc=auc(fpr,tpr)*100
model.fit (X_train, y_train)
lin_precision=tpr/(tpr+fpr)
cm=metrics.confusion_matrix(y_test,ypredict)
plt.figure(1)
plt.plot(fpr, tpr)




