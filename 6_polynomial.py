# Write a python program to implement polynomial regression for given dataset.


# Importing the libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
  
#Importing the dataset 
datas=pd.read_csv('C:\\Users\\Arbaaz\\Desktop\\CSV\\data1.csv')

print(datas)
print(datas.head())

# Dividing the dataset into 2 components
X = datas.iloc[:, 1:2].values 
y = datas.iloc[:, 2].values


# Fitting Linear Regression to the dataset 
from sklearn.linear_model import LinearRegression 
line = LinearRegression() 
  
line.fit(X, y)


# Visualising the Linear Regression results 
plt.scatter(X, y, color = 'blue') 
  
plt.plot(X,line.predict(X), color = 'red') 
plt.title('Linear Regression') 
plt.xlabel('Temperature') 
plt.ylabel('Pressure') 
  
plt.show()


# Fitting Polynomial Regression to the dataset 
from sklearn.preprocessing import PolynomialFeatures 
  
poly = PolynomialFeatures(degree = 8) 
X_poly = poly.fit_transform(X)

lin2 = LinearRegression()
lin2.fit(X_poly, y)
plt.scatter(X, y, color = 'blue')
  
plt.plot(X, lin2.predict(poly.fit_transform(X)), color = 'red')
plt.title('Polynomial Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
  
plt.show()

