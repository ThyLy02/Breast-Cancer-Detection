from src.model import BreastCancermodel
from src.utils import plot

import pandas as pd
#Load data
X_train = pd.read_csv("data/xtrain.csv", header=None)
Y_train = pd.read_csv("data/ytrain.csv", header=None)
X_test = pd.read_csv("data/xtest.csv", header=None)
Y_test = pd.read_csv("data/ytest.csv", header=None)


#Load model
classifier = BreastCancermodel().Loadmodel()
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ["accuracy"])

#Validate (predict on test set)
T = 0.5 #classification threshold
Y_pred = classifier.predict(X_test)
Y_pred = [ 1 if y>=0.5   else 0 for y in Y_pred ]

# get value in validation (test)
value_test = classifier.evaluate(X_test, Y_test)
print ("Loss in validation = %.4f" % value_test[0])
print ("accucary in validation = %.4f" % value_test[1])

plot(history)
