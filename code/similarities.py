from scipy.spatial import distance
from sklearn.metrics import pairwise
import scipy.stats
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine(*args):
    #print(pairwise.PAIRWISE_DISTANCE_FUNCTIONS)
    return 1- pairwise.pairwise_distances(*args, metric ='cosine')
    #return cosine_similarity(*args)


def msd(data):
    x,y = data.shape
    vals = data.values
    out = np.zeros((x,x))
    for i in range(x):
        first = vals[i]
        first = first - np.mean(first)
        first_sqr = [a**2 for a in first]
        meanf = np.mean(first)
        for j in range(x):
            second = vals[j]
            out[i][j]= 25 - np.mean((first - second)**2)
    return out

#pairwise.pairwise_distances(*args, metric='euclidean')


def pearson(data):
    x,y = data.shape
    vals = data.values
    out = np.zeros((x,x))
    #out_alt = np.zeros((x,x))
    for i in range(x):
        #first = vals[i]
        #first = first - np.mean(first)
        #first_sqr = [a**2 for a in first]
        for j in range(x):
            #second = vals[j]
            #second = second - np.mean(second)
            #second_sqr = [a ** 2 for a in second]
            #out[i][j]= np.dot(first,second)/np.sqrt(np.dot(first_sqr,second_sqr))
            out[i][j]= scipy.stats.pearsonr(vals[i], vals[j])[0]
            #out_alt[i][j]= scipy.stats.pearsonr(vals[i], vals[j])[0]
    #print(out)
    return out
