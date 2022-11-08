#Write a python program to implement simple linear regression to predict house price

# Importing Libraries
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

#Load the Boston Housing Data Set from sklearn.datasets and print it
#from sklearn.datasets import load_boston

boston = pd.read_csv('C:\\Users\\Arbaaz\\Desktop\\CSV\\housing.csv')
print("************ Priting dataset ************")
print(boston)


#Transform the data set into a data frame 
#NOTE: boston.data = the data we want, 
#      boston.feature_names = the column names of the data
#      boston.target = Our target variable or the price of the houses

#boston_data=pd.read_csv('housing.csv',usecols=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
    #'RAD', 'TAX', 'PTRATIO'])

boston_feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO']
df_x = pd.DataFrame(boston, columns = boston_feature_names)
df_y = pd.DataFrame(boston['MEDV'])


#Get some statistics from our data set, count, mean standard deviation etc.

df_x.describe()


#Initialize the linear regression model


#Split the data into 67% training and 33% testing data
#NOTE: We have to split the dependent variables (x) and the target or independent variable (y)

reg = linear_model.LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33)

#Train our model with the training data

reg.fit(x_train, y_train)

#Print the coefecients/weights for each feature/column of our model
print("********** Printing Coefficient weight for each column ***********")
print(reg.coef_)

#print our price predictions on our test data
y_pred = reg.predict(x_test)
print("*********** Printing prediction based on test **********")
print(y_pred)

#Print the the prediction for the third row of our test data actual price = 13.6
print("****** Printing prediction for third row of dataset ******")
print(y_pred[2])

#print the actual price of houses from the testing data set
print("********* Actual Price *******")
print(y_test)

# To check model performance/accuracy using,
# mean squared error which tells you how close a regression line is to a set of points.
print("************** Checking Accuracy of model **********")
print(np.mean((y_pred-y_test)**2))
