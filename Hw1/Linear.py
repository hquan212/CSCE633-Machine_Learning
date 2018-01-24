#!/usr/bin/python
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def import_data(dir):
    #Import the data set using a pandas frame with condition that area>0
    data = pd.read_csv(dir)
    data=data[data['area'] > 0]
    return data

#x is the features matrix, y is the output vector, w is the minimized weight vector
def OLS(x, y):
    w = np.dot(np.linalg.inv(np.dot(np.matrix.transpose(x), x)), np.dot(np.matrix.transpose(x),y))
    return w

def RSS(x_test,w,y_test):
    RSS = np.dot( (y_test - np.dot(x_test, w) ), (y_test - np.dot(x_test, w)))
    return RSS

def Nonlin(x):
    print "You are running the Non-linear training"
    return np.square(x)
#---------------------Run main function

#import into np arrays
training = import_data('train.csv')
testing = import_data('test.csv')

#get output matrix
y_train = np.array(training['area'])
y_test = np.array(testing['area'])

#get the features matrix
del training['area'], testing['area']

norm = training.apply(lambda x: np.sqrt(x**2).sum()/x.shape[0])
training /= norm
testing /= norm
x_train = np.array(training)
x_test = np.array(testing)

# ----------------------------#To make a non-linear model
x_train = Nonlin(x_train)

#-----------------------------

#our trained weights from the training data
w = OLS(x_train, y_train)

#Lets get our RSS!
print "The value of RSS is: ", RSS(x_test, w, y_test)