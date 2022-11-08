#Write a Python program to prepare scatter plot (use Forge/iris Dataset)

#importing libraries
import matplotlib.pyplot as plt
import pandas as pd


#loading data from csv file
dataset=pd.read_csv("C:\\Users\\Arbaaz\\Desktop\\CSV\\iris.csv")

# Creating 3 different dataframes for variety values
setosa=dataset[dataset['variety']=='Setosa']
virginica=dataset[dataset['variety']=='Virginica']
versicolor=dataset[dataset['variety']=='Versicolor']

print(dataset.describe())

fig,ax=plt.subplots(1,2,figsize=(9,9))

setosa.plot(x="sepal.length", y="sepal.width", kind="scatter",ax=ax[0],label='Setosa',color='r')
versicolor.plot(x="sepal.length",y="sepal.width",kind="scatter",ax=ax[0],label='Versicolor',color='b')
virginica.plot(x="sepal.length", y="sepal.width", kind="scatter", ax=ax[0], label='Virginica', color='g')

setosa.plot(x="petal.length", y="petal.width", kind="scatter",ax=ax[1],label='Setosa',color='r')
versicolor.plot(x="petal.length",y="petal.width",kind="scatter",ax=ax[1],label='Versicolor',color='b')
virginica.plot(x="petal.length", y="petal.width", kind="scatter", ax=ax[1], label='Virginica', color='g')


ax[0].set(title='Sepal Comparasion',  ylabel='sepal-width')
ax[1].set(title='Petal Comparasion',  ylabel='petal-width')

plt.show()
