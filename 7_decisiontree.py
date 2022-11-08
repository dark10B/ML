# Write python program to implement decision tree whether or not to play tennis

# importing data
import numpy as np
import pandas as pd
df = pd.read_csv('C:\\Users\\Arbaaz\\Desktop\\CSV\\weather.csv')

# converting categorical variables into dummies/indicator variables
df_getdummy=pd.get_dummies(data=df, columns=['Temperature', 'Outlook', 'Windy','Humidity'])


# separating the training set and test set
from sklearn.model_selection import train_test_split

X = df_getdummy.drop('Played?',axis=1)
y = df_getdummy['Played?']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)



# visualising the decision tree diagram

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=3)
dtree.fit(X_train,y_train)

predictions = dtree.predict(X_test)
print(predictions)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6,6))
plot_tree(dtree, feature_names=df_getdummy.columns, fontsize=6, filled=True, class_names=['Not Play', 'Play'])
plt.show()
