from src.model import BreastCancermodel

import pandas as pd



#Load data
X_train = pd.read_csv("data/xtrain.csv", header=None)
Y_train = pd.read_csv("data/ytrain.csv", header=None)
X_test = pd.read_csv("data/xtest.csv", header=None)
Y_test = pd.read_csv("data/ytest.csv", header=None)
#Load model
classifier = BreastCancermodel().Loadmodel()
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ["accuracy"])

#train
history=classifier.fit(X_train, Y_train, batch_size = 1, epochs = 20 )

#summary
classifier = BreastCancermodel().summary_model()





