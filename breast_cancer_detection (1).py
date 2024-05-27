# -*- coding: utf-8 -*-
"""Breast_cancer_detection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ymE_oVeuCkH3y3PcbQTWIg1p5ztuDtvt
"""

import sklearn.datasets

import numpy as np

"""
#Data Loading
"""

breast_cancer = sklearn.datasets.load_breast_cancer()

X = breast_cancer.data
Y = breast_cancer.target

print(X)
print(Y)

print(X.shape,Y.shape)

import pandas as pd
data=pd.DataFrame(X , columns=breast_cancer.feature_names)

data['class']=Y

data.head(5)

print(data['class'].value_counts())
print("Mean of the data : ",data['class'].mean())

print(breast_cancer.target_names)

"""The above shows  1 is malignant and 2 benign i.e with cancer there are more samples than without"""

data.groupby('class').mean()

"""#Train - Test split"""

from sklearn.model_selection import train_test_split

X = data.drop('class',axis=1)
Y=data['class']

"""x and y are now dataframes and not numpy"""

type(X)

X_train, X_test, Y_train,Y_test = train_test_split(X,Y,test_size = 0.1,stratify = Y, random_state = 42)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)

print(Y_train.mean(),Y_test.mean())

"""#Data Visualization"""

import matplotlib.pyplot as plt

plt.plot(X_train.T,'*')
plt.xticks(rotation = 'vertical')
plt.show()

"""#Binarizing the data"""

X_binarised_train = X_train.apply(pd.cut, bins = 2, labels =[1,0])

plt.plot(X_binarised_train.T,'*')
plt.xticks(rotation = 'vertical')
plt.show()

X_binarised_test = X_test.apply(pd.cut,bins=2,labels = [1,0])

type(X_binarised_test)

X_binarised_test = X_binarised_test.values
X_binarised_train = X_binarised_train.values

type(X_binarised_train)

"""#MP neuron"""

X_binarised_train.shape

for b in range(X_binarised_train.shape[1]+1):
  Y_pred_train = []
  accurate_rows = 0
  for x,y in zip(X_binarised_train,Y_train):
    Y_pred = (np.sum(x)>=b)
    Y_pred_train.append(Y_pred)
    accurate_rows+=(y == Y_pred)
  print(b,accurate_rows/X_binarised_train.shape[0])

"""accuracy is max at b=27

#Evaluation
"""

from sklearn.metrics import accuracy_score

Y_test.shape

b=27
Y_pred_test = []
for x in X_binarised_test:
  Y_pred = (np.sum(x)>=b)
  Y_pred_test.append(Y_pred)

accuracy = accuracy_score(Y_pred_test, Y_test)
print(accuracy)

"""#MP neuron Class"""

class MpNeuron:

    def __init__(self):
        self.b = None

    def model(self, x):
        return (sum(x) >= self.b)

    def predict(self, X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)
    def fit(self, X, Y):
        accuracy = {}
        for b in range(X.shape[1] + 1):
            self.b = b
            Y_pred = self.predict(X)
            accuracy[b] = accuracy_score(Y, Y_pred)
        best_b = max(accuracy, key=accuracy.get)
        self.b = best_b

        print('Optimal value of b is ', best_b)
        print('Highest accuracy is ', accuracy[best_b])

mp_neuron=MpNeuron()

Y_train.shape

mp_neuron.fit(X_binarised_train,Y_train)

mp_neuron.fit(X_binarised_test,Y_test)

"""#Perceptron model

"""

class Perceptron:

  def __init__(self):
    self.w = None
    self.b = None

  def model(self, x):
    return 1 if (np.dot(self.w, x) >= self.b) else 0

  def predict(self, X):
    Y = []
    for x in X:
      result = self.model(x)
      Y.append(result)
    return np.array(Y)

  def fit(self, X, Y,epochs,lr):
    self.w = np.ones(X.shape[1])
    self.b = 0
    for i in range(epochs):
      for x, y in zip(X, Y):
        y_pred = self.model(x)
        if y == 1 and y_pred == 0:
          self.w += lr*x
          self.b += lr*1
        elif y == 0 and y_pred == 1:
          self.w -= lr*x
          self.b -= lr*1

perceptron = Perceptron()

perceptron.fit(X_train,Y_train,2,1)

Y_predicted = perceptron.predict(X_train)
accuracy = accuracy_score(Y_predicted,Y_train)
print(accuracy)

Y_predicted = perceptron.predict(X_test)
accuracy = accuracy_score(Y_predicted,Y_test)
print(accuracy)

