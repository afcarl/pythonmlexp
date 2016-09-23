# -*- coding: utf-8 -*-
#title              : vq.py
#author             : Victor Mutai
#date               : 20160923
#version            : 0.1
#license            : GNU
"""
Vector Quantization (VQ):
    https://www.willamette.edu/~gorr/classes/cs449/Unsupervised/competitive.html
    
    1. Choose the number of clusters M
    2. Initialize the prototypes w_1, ..., w_m (e.g. choose M random vectors from input data)
    3. Repeat until stopping criterion is satisfied 
        * Randomly pick an input x
        * Determine the "winning" node k by finding the prototype vector that satisfies
            | w_k - x | <= |w_i - x| (for all i)
            note: if the prototypes are normalized, this is eq to maximizing w_i x
        * Update only the winning prototype weights according to
            w_k(new) = w_k(old) + nu*(x - w_k(old)) 
            
            https://en.wikibooks.org/wiki/Artificial_Neural_Networks/Competitive_Learning
            
            This is called the std competitive learning rule
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate data 
mu1 =[0, 0]
mu2 =[3, 5]
mu3 =[-5, 3]
cov = [[1, 0], [0, 1]] # diagonal covariance

g1 = np.random.multivariate_normal(mu1, cov, 50)
g2 = np.random.multivariate_normal(mu2, cov, 50)
g3 = np.random.multivariate_normal(mu3, cov, 50)
g4 = np.concatenate((g1, g2))

data = np.concatenate((g4, g3))
pdata = data.T
  
plt.plot(pdata[0], pdata[1], 'x')
plt.axis('equal')


# Vector Quantization Algorithm

M = 3 # choose no of clusters
Nu = 0.2 # Competitive learning rate
np.random.shuffle(data) # Randomize data
W = data[0:M] # init prototypes from random data

# Repeat until stopping criterion is satisfied
for i in range(1, 1000):
    indx = np.random.randint(0,len(data)-1) # pick a random index
    x = data[indx] # random x
    
    dist = np.linalg.norm(W[0] - x) # euclidean distance from first w to x
    k = 0 # winning index
    for p in range(1,M):
        d = np.linalg.norm(W[p] - x)
        if d < dist: # find the least distance
            dist = d
            k = p # update winning index
             
    W[k] = W[k] + Nu*(x - W[k]) # update winning prototype
    
WT = W.T 
plt.plot(WT[0], WT[1], 'o') # add prototypes to the plot
plt.show() # show plot.


"""
    TO DO: update the plot in realtime to see how the prototypes change
"""


        


