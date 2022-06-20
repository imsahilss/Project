from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import minmax_scale  
import numpy
import os
import pandas as pd  
from collections import deque

class MachineLearningAlgo:
    def __init__(self):

        #self.clf = svm.SVC(kernel="linear")
        #print("Using SVC Algo") 

        #Decision Tree
        #self.clf = tree.DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
        #print("Using DecisionTreeClassifier Algo") 


        #Forests of randomized trees
        #self.clf = RandomForestClassifier(n_estimators=10)
        #print("Using RandomForestClassifier Algo") 


        self.clf = MLPClassifier(hidden_layer_sizes=(7), activation="logistic", solver='sgd', beta_1=0.9, beta_2=0.9,
                                 learning_rate="constant", learning_rate_init=0.1, momentum=0.9)
        print("Using MLPClassifier Algo") 


        X_train = pd.read_csv('result.csv')
        y_train = X_train["type"]
        del X_train["type"]
        X_train.iloc[:] = minmax_scale(X_train.iloc[:])
        self.clf.fit(X_train, y_train.values.ravel())  
        

    def classify(self, data):      
        prediction = self.clf.predict(data)
        print("prediction result ", prediction)
        return prediction
