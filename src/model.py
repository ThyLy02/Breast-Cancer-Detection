# Building model
from keras.models import Sequential
from keras.layers import Dense

class BreastCancermodel:
    def __init__(self):

        self.classifier = Sequential() # Initialising the ANN creates a blank model
        self.classifier.add(Dense(units = 16, activation = 'relu', input_dim = 30)) # create layer; units =16 (nodes) is input dim= 30
        self.classifier.add(Dense(units = 8, activation = 'relu'))# 70% use relu, keep the value and remove the negative value; hidden layer
        self.classifier.add(Dense(units = 6, activation = 'relu'))
        self.classifier.add(Dense(units = 1, activation = 'sigmoid')) # To break the result, we also classify 2 classes ; output layer

    def Loadmodel(self):
        return self.classifier
    def summary_model(self):
        return self.classifier.summary()


# Having 4 layers