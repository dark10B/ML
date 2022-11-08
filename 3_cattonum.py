# Write a python program to encode categorical values into numeric values for given dataset

#importing libraries
import pandas as pd

# Loading dataset from csv file
dataset=pd.read_csv('C:\\Users\\Arbaaz\\Desktop\\CSV\\iris.csv')

# Converting column to category using pandas
dataset["variety"]=dataset["variety"].astype('category')
print("******* Data types of fields in iris.csv ******")
print(dataset.dtypes)
print(dataset["variety"])

# Assigning encoded variable to new column using cat.codes
print("***** Adding new column ******")
dataset['variety_num']=dataset['variety'].cat.codes
print(dataset.dtypes)
print()
print("_______________________________________________________")

# To view full dataset
pd.set_option('display.max_rows',150)
pd.set_option('display.max_columns',7)
print(dataset)
