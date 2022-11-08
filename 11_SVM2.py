#import pandas as pd
#import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
 
# IRIS Data Set
 
iris = datasets.load_iris()
X = iris.data
y = iris.target
 
# Creating training and test split
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify = y)
 
# Feature Scaling
 
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
 
# Training a SVM classifier using SVC class
svm = SVC(kernel= 'linear', random_state=1, C=0.1)
svm.fit(X_train_std, y_train)
 
# Mode performance
from sklearn import metrics
y_pred = svm.predict(X_test_std)
print('Accuracy: %.3f' % (metrics.accuracy_score(y_test, y_pred)*100))
