# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 13:11:33 2020

@author: Madhu
"""
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

df=pd.read_csv('C:/Users/Madhu/Downloads/train.csv')
selected_data=df[['PassengerId','Survived','Pclass','SexID','Age','SibSp','Parch']]
#data wrangling
selected_data.isnull()
selected_data.isnull().sum()

#fix Age
selected_data['Age'].fillna((selected_data['Age'].mean()), inplace=True)

#remove the other selected nullvalues
selected_data.dropna(inplace=True)

#plot 
sns.countplot(x='Survived',data=selected_data)
sns.barplot('Pclass','Survived',data=selected_data,color="darkturquoise")

#seperate features
selected_data_X=selected_data[['PassengerId','Pclass','SexID','Age','SibSp','Parch']]
selected_data_Y=selected_data[['Survived']]

#split into test and train set
X_train, X_test, Y_train, Y_test= train_test_split(selected_data_X,selected_data_Y,test_size=0.20)

#create the object
from sklearn.linear_model import LogisticRegression
logistic_regression= LogisticRegression()

#fit into model
logistic_regression.fit(X_train,Y_train)

Y_pred= logistic_regression.predict(X_test)
print(Y_pred)

#matrix
from sklearn.metrics import  confusion_matrix
confusion_matrix(Y_test,Y_pred)

#accuracy calculation
from sklearn import metrics 
metrics.accuracy_score( Y_test, Y_pred)

