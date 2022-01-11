#Data PreProcessing

#importing the libraries 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from pandas._libs import missing

#importing the dataset

dataset = pd.read_csv('Data.csv')
#print(dataset)
#creatin a matrix out of the dataset
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# #Take care of missing datas

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NaN,strategy = 'mean' )
imputer = imputer.fit(x[:, 1:3])
x[:,1:3] = imputer.transform(x[:,1:3])
# print(x)

#Feature SCALING

""" from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
# print(x_train)
# print(x_test) """

#Encoding Categorical data 

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])
print(x)
onehotencoder = OneHotEncoder()
x = onehotencoder.fit_transform(x[:,0].reshape(-1,1)).toarray()   
print(x)
   #DummyEncoding using pandas
   
# z= pd.get_dummies(x[:,0])
# print(z)
# labelencoder_y = LabelEncoder()
# y = labelencoder_y.fit_transform(y)
# print(y)

#Splitting the data set into training data and test data

from sklearn.model_selection import train_test_split 
x_train , x_test , y_train ,_y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)
#print(x_train)
