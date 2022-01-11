#Polynomial Regression 
#importing the libraries 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from pandas._libs import missing

#importing the dataset

dataset = pd.read_csv('Position_Salaries.csv')
#print(dataset)
#creatin a matrix out of the dataset
x = dataset.iloc[:,1:2].values
#print(x)
y = dataset.iloc[:, 2:3].values
#print(y)


""" #Splitting the data set into training data and test data

from sklearn.model_selection import train_test_split 
x_train , x_test , y_train ,y_test = train_test_split(x,y, test_size = 0.5, random_state = 0) """

#Linear regression model to compare

from sklearn.linear_model import LinearRegression
Lregressor = LinearRegression()
Lregressor.fit(x,y)

#Polynomial regression model fitting to the dataset 

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
x_poly = poly_reg.fit_transform(x)
linear_reg = LinearRegression()
linear_reg.fit(x_poly,y)
#print(x_poly)

# #visualizing the linear regression model 
# plt.scatter(x,y,color='red')
# plt.plot(x,Lregressor.predict(x),color='blue')
# plt.title('Linearregression model  bluff ')
# plt.xlabel('Level')
# plt.ylabel('Salary')
# plt.show()
# #visualizing the polynomial regression model
# x_grid = np.arange(min(x),max(x),0.1)
# x_grid = x_grid.reshape(len(x_grid),1)
# plt.scatter(x,y,color='red')
# plt.plot(x_grid,linear_reg.predict(poly_reg.fit_transform(x_grid)),color='blue')
# plt.title('Linearregression model  bluff ')
# plt.xlabel('Level')
# plt.ylabel('Salary')
# plt.show()

q = np.array(6.5)
#predicting linear regression model

Lregressor.predict(q.reshape(-1,1))

 #Predicting by Polynomial regression model
linear_reg.predict(poly_reg.fit_transform(q.reshape(-1,1)))

