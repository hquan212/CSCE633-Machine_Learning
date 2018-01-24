#!/usr/bin/python
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math

#----------------testing point, data locations, training data, k
def knearestneighbor(p, x, v, k):
    assert(len(v) > k)
    x = x.astype(float)
    p = np.array(p).astype(float)
    norm = np.linalg.norm(x, axis=0)
    x[:] /= norm
    p[:] /= norm
    distance = np.linalg.norm(p[:] - x[:], ord=2, axis=1)
    voters = v.iloc[np.argsort(distance)[:k]]
    return(voters['class'].mode())

def concatrain(str):
    return np.expand_dims(training[str].values, 0)

def concatest(str):
    return np.expand_dims(testing[str].values, 0)
#---------------------import data function
def import_data(dir):
    data = pd.read_csv(dir)
    data['class'] = (data['area'] > 0).astype(int)
    return data

#--------------------Main
print("Enter how many neighbors should we test with (choose an odd k value) :")
k = input()

#---------------------Import the training set
training = import_data('train.csv')
location = np.transpose(np.concatenate([concatrain('X'), concatrain('Y'),concatrain('month'),
  concatrain('day'), concatrain('FFMC'),concatrain('DMC'),concatrain('DC'),concatrain('ISI'),concatrain('temp'),
    concatrain('RH'),concatrain('wind'), concatrain('rain')]))

#--------------------Write the function for the knearestneighbor
nearest = lambda x: knearestneighbor(x, location, training, k)


print location
#---------------------Import the testing set
testing = import_data('test.csv')
testing_set = np.transpose(np.concatenate([concatest('X'), concatest('Y'),concatest('month'),
  concatest('day'), concatest('FFMC'),concatest('DMC'),concatest('DC'),concatest('ISI'),concatest('temp'),
    concatest('RH'),concatest('wind'), concatest('rain')]))

#---------------------Lets use the KNN on our testing set
knn_values = [np.asscalar(nearest(test)) for test in testing_set]
real_test = testing['class']

#----------------------Cross validation
correct = (np.array(real_test) == np.array(knn_values)).astype(int)

print "Correct predictions", np.sum(correct)
print "Accuracy ", np.mean(correct)*100

# plt.scatter(RH, temp, alpha=0.75, color='purple')
# plt.grid(True)
# plt.title("Relative Humidity vs Temp")
# plt.ylabel("RH, relative humidity")
# plt.xlabel("Temp, C")
# plt.show()