#Multiple_Linear_ regresision

#importing the libraries 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from pandas._libs import missing

#importing the dataset

dataset = pd.read_csv('50_Startups.csv')
#print(dataset)
#creatin a matrix out of the dataset
x = dataset.iloc[:, :-1].values
#print(x)
y = dataset.iloc[:, 4].values
#print(y)


#Devraj Encoding 
# onehotencoder = OneHotEncoder(dtype=int)
# x_coded = onehotencoder.fit_transform(x[:,3].reshape(-1,1)).toarray() #gives array of (50,3) with dummies
# dfOneHot = pd.DataFrame(x_coded, columns = [i for i in range(len(dataset['State'].unique()))]) #gives new dataset with x_coded data

# new_dataset = pd.concat([dataset, dfOneHot], axis=1) #adding dfOneHot dataset to our original dataset
# new_dataset = new_dataset.drop(['State'], axis=1) 

# print(new_dataset)

# #Encoding the categorical data 

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
new= ohe.fit_transform(dataset[['State']]).toarray()  #Provides the dummies for the state column
#print(new)
x=np.hstack((new,dataset[['R&D Spend','Administration','Marketing Spend']].values,)) # adds the dummies to the x horizontally 
#print(x)

#avoiding the dummy variable trap

x = x[:,1:]
# print(x)



#Splitting the data set into training data and test data

from sklearn.model_selection import train_test_split 
x_train , x_test , y_train ,y_test = train_test_split(x,y, test_size = 0.5, random_state = 0)
#print(x_train)

# # #Feature SCALING

# # """ from sklearn.preprocessing import StandardScaler
# # sc_x = StandardScaler()
# # x_train = sc_x.fit_transform(x_train)
# # x_test = sc_x.transform(x_test)
# # # print(x_train)
# # # print(x_test) """

#Fitting mULTIOLE lINEARrEGRESSION 

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predicting the Test Set Results 

y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

""" #visualizing the dataset 

# plt.scatter(x_train[:,:-4],y_train,color= 'red')
# #plt.plot(x_test,regressor.fit(x_train,y_train),color='blue')
# plt.show() """

#Building ana optimal model using backward elimination

import statsmodels.api as sm 
x = np.append(arr = np.ones((50,1)).astype(int),values = x,axis = 1)  # Adding ones to the x as per multiple regression model see note 
# print(x)
x_opt = x[:,[0,1,2,3,4,5]]
#print(x) 
regressor_OLS = sm.OLS(y,x_opt).fit()
regressor_OLS.summary()

#Removing the x2 variable as the pvalue of x2 is more than significance level 0.05

x_opt = x[:,[0,1,3,4,5]]
#print(x) 
regressor_OLS = sm.OLS(y,x_opt).fit()
regressor_OLS.summary()

#Removing the x1 variable as the pvalue of x2 is more than significance level 0.05

x_opt = x[:,[0,3,4,5]]
#print(x) 
regressor_OLS = sm.OLS(y,x_opt).fit()
regressor_OLS.summary()

##Removing the x2 variable as the pvalue of x2 is more than significance level 0.05

x_opt = x[:,[0,3,5]]
#print(x) 
regressor_OLS = sm.OLS(y,x_opt).fit()
regressor_OLS.summary()

##Removing the x2 variable as the pvalue of x2 is more than significance level 0.05

x_opt = x[:,[0,3]]
#print(x) 
regressor_OLS = sm.OLS(y,x_opt).fit()
regressor_OLS.summary()

#predicting the y value 
x_new = x_test[:,[0,3]]
y_val_ML = regressor_OLS.predict(x_new)
print(y_val_ML)

plt.scatter(x_test[:,4],y_test,color = 'red')
plt.scatter(x_test[:,4],y_pred,color='blue')
plt.scatter(x_test[:,4],y_val_ML,color = 'green')

