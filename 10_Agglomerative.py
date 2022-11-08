# Write a python program to implement Agglomerative clustering on a synthetic dataset.

# Importing the libraries  
import numpy as np  
import matplotlib.pyplot as mtp  
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


# Generating synthetic dataset and dividing them into two parts
X= -2 *np.random.rand(100,2)
X1 = 1 + 2 * np.random.rand(50,2)
X[50:100, :] = X1

#Finding the optimal number of clusters using the dendrogram  
import scipy.cluster.hierarchy as shc  
dendro = shc.dendrogram(shc.linkage(X, method="single"))  
mtp.title("Dendrogram Plot")  
mtp.ylabel("X1")  
mtp.xlabel("X")  
mtp.show()

#training the hierarchical model on dataset  
hc= AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='single')  
y_pred= hc.fit_predict(X)

#visulaizing the clusters  
mtp.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s = 100, c = 'blue', label = 'Cluster 1')  
mtp.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s = 100, c = 'green', label = 'Cluster 2')  
mtp.scatter(X[y_pred== 2, 0], X[y_pred == 2, 1], s = 100, c = 'red', label = 'Cluster 3')  
mtp.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')  
mtp.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')  

mtp.title('Clusters of data')  
mtp.xlabel('X Values')  
mtp.ylabel('X1 Values')  
mtp.legend()  
mtp.show()  

