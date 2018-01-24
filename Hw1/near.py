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
accuracies = []
kvals = []
best = []

for k in range(0,15):
    #---------------------Import the training set
    print "Starting with K =", 2*k +1
    training = import_data('train.csv')
    location = np.transpose(np.concatenate([concatrain('X'), concatrain('Y'),concatrain('month'),
    concatrain('day'), concatrain('FFMC'),concatrain('DMC'),concatrain('DC'),concatrain('ISI'),concatrain('temp'),
        concatrain('RH'),concatrain('wind'), concatrain('rain')]))

    t1,t2,t3,t4 = location[:150], location[150:250], location[250:350], location[350:]
    y1,y2,y3,y4 = training['class'],training['class'],training['class'],training['class']


    #--------------------Write the function for the knearestneighbor
    nearest1 = lambda x: knearestneighbor(x, t1, training, 2*k+1)
    nearest2 = lambda x: knearestneighbor(x, t2, training, 2*k+1)
    nearest3 = lambda x: knearestneighbor(x, t3, training, 2*k+1)
    nearest4 = lambda x: knearestneighbor(x, t4, training, 2*k+1)


    #---------------------Import the testing set
    testing = import_data('test.csv')
    testing_set = np.transpose(np.concatenate([concatest('X'), concatest('Y'),concatest('month'),
    concatest('day'), concatest('FFMC'),concatest('DMC'),concatest('DC'),concatest('ISI'),concatest('temp'),
        concatest('RH'),concatest('wind'), concatest('rain')]))

    #---------------------Lets use the KNN on our testing set
    knn_values1 = [np.asscalar(nearest1(test)) for test in testing_set]
    knn_values2 = [np.asscalar(nearest2(test)) for test in testing_set]
    knn_values3 = [np.asscalar(nearest3(test)) for test in testing_set]
    knn_values4 = [np.asscalar(nearest4(test)) for test in testing_set]
    real_test = testing['class']

    #----------------------Cross validation
    correct = (np.array(real_test) == np.array(knn_values1)).astype(int)
    print "Accuracy for set 1", np.mean(correct)*100
    accuracies.append(np.mean(correct)*100)

    correct = (np.array(real_test) == np.array(knn_values2)).astype(int)
    print "Accuracy for set 2", np.mean(correct)*100
    accuracies.append(np.mean(correct)*100)

    correct = (np.array(real_test) == np.array(knn_values3)).astype(int)
    print "Accuracy for set 3", np.mean(correct)*100
    accuracies.append(np.mean(correct)*100)

    correct = (np.array(real_test) == np.array(knn_values4)).astype(int)
    print "Accuracy for set 4", np.mean(correct)*100
    accuracies.append(np.mean(correct)*100)
    kvals.append(2*k+1)
    best.append(max(accuracies))
    print 

plt.scatter(kvals,best,s=None)
plt.title("Accuracies for different values of K using cross validation")
plt.ylim(0,100)
plt.xlabel("K Nearest Neighbors")
plt.ylabel("Accuracies on KNN Algorithm (%)")
plt.grid(True)
plt.show()