# Write a python program to implement k-means algorithm on a synthetic dataset.

# Importing Libraies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generating dataset and dividing them into two parts
X= -2 * np.random.rand(100,2)
X1 = 1 + 2 * np.random.rand(50,2)
X[50:100, :] = X1

# Plotting datapoints using scatter plot 
plt.scatter(X[ : , 0],X[ :, 1], s = 50,c='b')
plt.show()

# Processing generated data points
Kmean = KMeans(n_clusters=2)
Kmean.fit(X)

# Displaying centroids
print(Kmean.cluster_centers_)

# Display the cluster centroids (using green and red color)in graph
plt.scatter(X[ : , 0], X[ : , 1], s =50, c='b')
plt.scatter(-0.94665068, -0.97138368, s=200, c='g', marker='s')
plt.scatter(2.01559419, 2.02597093, s=200, c='r', marker='s')
plt.show()

# Displaying class labels of data points
print(Kmean.labels_)

# Predicting class for new datapoint
sample_test=np.array([-3.0,-3.0])
second_test=sample_test.reshape(1, -1)#Changing dimention
print(Kmean.predict(second_test))
