#Simple Linear Regression

#importing the libraries 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from pandas._libs import missing

#importing the dataset

dataset = pd.read_csv('Salary_Data.csv')
# print(dataset)
#creatin a matrix out of the dataset
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
#print(y)

#Splitting the data set into training data and test data

from sklearn.model_selection import train_test_split 
x_train , x_test , y_train ,y_test = train_test_split(x,y, test_size = 1/3, random_state = 0)
#print(x_train)

#Feature SCALING

""" from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
# print(x_train)
# print(x_test) """

#Fitting the simple regression model to the training set 

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predicting the test set results 

y_pred  = regressor.predict(x_test)
# print(y_pred)
# print(y_test)

#Visualizing the training set results

plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title('Salara Vs Experienc(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the test set results

plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title('Salara Vs Experienc(Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()