#Importing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# load the iris dataset
from sklearn.datasets import load_iris
iris = load_iris()

# store the feature matrix (X) and response vector (y)
X = iris.data
y = iris.target

# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Feature Scaling
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
 
# Training a SVM classifier using SVC class
svm = SVC(kernel= 'linear', random_state=1, C=0.1)
svm.fit(X_train_std, y_train)

# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics
y_pred = svm.predict(X_test_std)
print("SVM model accuracy(in %): ",metrics.accuracy_score(y_test, y_pred)*100)
