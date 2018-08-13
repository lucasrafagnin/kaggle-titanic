# coding: utf-8

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

train = pd.read_csv('../train.csv')

features = train.drop('Survived', axis=1)
outcomes = train['Survived']

features = pd.get_dummies(features)
features = features.fillna(0.0)

model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=6, min_samples_split=5)

X_train, X_test, y_train, y_test = train_test_split(features, outcomes, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('The training accuracy is', train_accuracy)
print('The test accuracy is', test_accuracy)
