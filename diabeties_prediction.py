import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

diabeties = pd.read_csv("diabetes.csv")
# print(diabeties)
# print(diabeties.isnull().sum())
# print(diabeties.columns)

x = diabeties.iloc[:,:-1]
y = diabeties.iloc[:,-1]
# print(x)
# print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LogisticRegression
regression = LogisticRegression()
regression.fit(X_train,y_train)

from sklearn.model_selection import cross_val_score
mes = cross_val_score(regression,x,y,scoring="accuracy",cv =15)

pred = regression.predict(X_test)
print(pred)

print(regression.score(X_test,y_test))
from sklearn.metrics import r2_score 
score = r2_score(pred,y_test)
print(score)