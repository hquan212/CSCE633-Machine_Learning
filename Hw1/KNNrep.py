#!/usr/bin/python
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt

#----------------Def hamming distance

def hamming(p, x, v, k):
    pass

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


#---------------------import data function
def import_data(dir):
    data = pd.read_csv(dir)
    data['class'] = (data['area'] > 0).astype(int)
    return data

#--------------------Historgram

def plotloghist(array):
    log = np.log(array)
    print log
    plt.hist(log,alpha=0.75,color='green')
    plt.grid(True)
    plt.title("Histogram of fires per month")
    plt.xlabel("Frequency of fires")
    plt.ylabel("Months of year")
    plt.show()

def concatrain(file, str):
    return np.expand_dims(file[str].values, 0)
