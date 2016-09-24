# -*- coding: utf-8 -*-
#title              : brownian.py
#author             : Victor Mutai
#date               : 20160923
#version            : 0.1
#license            : GNU
"""

Brownian Motion:
    
    Brownian motion over [0,T] is a r.v. W(t) that depends continuously
    on t elem_of [0,T] and satisfies the following three conditions:
    1. W(0) = 0 (with probability 1)
    2. For 0<=s<t<=T ther r.v. given by the incremnt W(t)-W(s) is
        normally distr' with mean 0 and variance t-s, equivalently,
        W(t)-W(s)~=sqrt(t-s)*N(0,1), where N(0,1) denotes norm' dist'
        r.v. with 0 mean and unit variance.
    3. For 0<=s<t<u<v<=T the incrememts W(t)-W(s) and W(v)-W(u) are independent
    
    
    Algorithm:
    
    1. Set time interval [0, T]
    2. Set number of steps (N) to compute in [0, T]
    3. Compute the time step (dt)
    4. Generate array of brownian movements (dW)
    5. Sum the array cummulatively (W)
    

"""

import numpy as np
import matplotlib.pyplot as plt

T = 1.
N = 1000
dt = T/N
dW = np.sqrt(dt)*np.random.randn(1,N)
W = np.cumsum(dW)
t = np.arange(0,T+dt,dt)
W = np.append(np.array([0]), W)

plt.plot(t, W, '-')
plt.show()
