# Write a Python program to find null values in given dataset and remove them

# importing Libraries
import pandas as pd

# Reading dataset
dataset = pd.read_csv('C:\\Users\\Arbaaz\\Desktop\\CSV\\employees.csv')

print(dataset.describe())
print(dataset)

dataset.dropna(inplace=True)

print(dataset)
