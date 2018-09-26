import numpy as np
import math
#import cmath
#import matplotlib.pyplot as plt
#from scipy import signal
#from scipy.optimize import least_squares
#from scipy.optimize import minimize
#from scipy.io import loadmat
import random
from numpy import linalg as LA
from fun1d import *
import time

def sample_poisson_interval(t, l):
    # get a sample of poisson process
    #initialization
    x = np.zeros(1) # time interval
    t_sum = np.zeros(1) # time of jumps
    
    #sample time of jumps
    while t_sum[-1] < t[-1]:
        u = - np.log(np.random.uniform(0,1,1)) / l
        x = np.append(x, u) 
        t_sum = np.append(t_sum, t_sum[-1] + u) 
    return t_sum

def sample_compound_poisson(t, l, *args):
    t_sum = sample_poisson_interval(t, l)
    
    y = np.zeros(len(t))
    ind_left = 0
    rtype = args[0]
    s = 0
    #sample signal
    for i in range(len(t_sum) - 1):
        ind_right = np.max(np.where(t < t_sum[i + 1])) + 1
        if ind_right > ind_left:
            if rtype == 'gaussian':
                mu = args[1]
                sigma = args[2]
                a = mu + sigma * np.random.randn(1)
            elif rtype == 'uniform':
                m = args[1]
                a = m * np.random.random(1)
            elif rtype == 'flip_coin':
                m = args[1]
                p = np.random.random(1)
                if p > 1/2:
                    a = m
                else:
                    a = - m
            elif rtype == 'const':
                a = args[1]
            s += a
            y[ind_left:ind_right] = s
        ind_left = ind_right
    return y

def prod_poisson_scat_coeff(t, l, nsample, g_hat, *args):
    n = t.shape[0]
    nw = g_hat.shape[1]
    y = np.zeros((n, nsample))
    f = np.zeros((nw + 2, nsample))
    arg = args
    for i in range(nsample):
        y[:, i] = sample_compound_poisson(t, l, *args)
        dy = np.diff(y[:,i], axis = 0)
        f[:, i] = scat_coeff(dy, g_hat)
    mu = np.mean(f, axis = 1)
    mu = mu[:, np.newaxis]
    sigma = np.mean((f - mu)**2, axis = 1)
    mu = np.squeeze(mu)
    f_new = mu + np.multiply(np.random.randn(nw + 2),  np.sqrt(sigma))
    return f_new
