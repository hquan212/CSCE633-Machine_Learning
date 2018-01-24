#!/usr/bin/python
import KNNrep
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt

accuracies = []
kvals = []

for k in range(0,15):

    #---------------------Import the testing set
    testing = KNNrep.import_data('test.csv')
    testing_set = np.transpose(np.concatenate([KNNrep.concatrain(testing,'X'), KNNrep.concatrain(testing,'Y'),KNNrep.concatrain(testing,'month'),
               KNNrep.concatrain(testing,'day'), KNNrep.concatrain(testing,'FFMC'),KNNrep.concatrain(testing,'DMC'),KNNrep.concatrain(testing,'DC'),
               KNNrep.concatrain(testing,'ISI'),KNNrep.concatrain(testing,'temp'),KNNrep.concatrain(testing,'RH'),KNNrep.concatrain(testing,'wind'), KNNrep.concatrain(testing,'rain')]))

    training = KNNrep.import_data('train.csv')
    location = np.transpose(np.concatenate([KNNrep.concatrain(training,'X'), KNNrep.concatrain(training,'Y'),KNNrep.concatrain(training,'month'),
               KNNrep.concatrain(training,'day'), KNNrep.concatrain(training,'FFMC'),KNNrep.concatrain(training,'DMC'),KNNrep.concatrain(training,'DC'),
               KNNrep.concatrain(training,'ISI'),KNNrep.concatrain(training,'temp'),KNNrep.concatrain(training,'RH'),KNNrep.concatrain(training,'wind'), KNNrep.concatrain(training,'rain')]))
    nearest =  lambda x: KNNrep.knearestneighbor(x, location, training, (2*k+1))
    # classifier = KNNrep.KnnClassifier('train.csv', k)
    # knn_values = [np.asscalar(classifier(test)) for test in testing_set]

    #---------------------Lets use the KNN on our testing set
    knn_values = [np.asscalar(nearest(test)) for test in testing_set]
    real_test = testing['class'].values

    #----------------------Cross validation
    correct = (np.array(real_test) == np.array(knn_values)).astype(int)

    #print "Correct ", np.sum(correct)
    accuracies.append(np.mean(correct)*100)
    kvals.append(2*k+1)
    #print "Accuracy ", np.mean(correct)*100

plt.scatter(kvals,accuracies,s=None)
plt.title("Accuracies for different values of K incl temp")
plt.ylim(0,100)
plt.xlabel("K Nearest Neighbors")
plt.ylabel("Accuracies on KNN Algorithm (%)")
plt.grid(True)
plt.show()