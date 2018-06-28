import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.io import loadmat
import random
from numpy import linalg as LA
import time
# from numbapro import vectorize

def gaussian(x):
    pi = math.pi
    y = 1 / np.sqrt(2 * pi) * np.exp(- x**2 / 2)
    return y

def window_filter_1d(n, s, xi):
    # generate one 1D gabor wavelet 
    x = np.arange(n) - np.floor(n/2)
    chi = np.zeros(n)
    chi = gaussian(x/s)
    
    o = np.exp(1j * xi * x)
    
    psi = np.multiply(chi, o)
    
    psi_hat = np.fft.fft(psi)
    return psi, psi_hat

def window_filter_family_1d(n, s, xi):
    # generate a family of 1D gabor wavelets with specified scales and rotations in space
    ns = s.shape[0]
    nxi = xi.shape[0]
    
    psi = np.zeros((n, ns, nxi),dtype=complex)
    psi_hat = np.zeros((n, ns, nxi),dtype=complex)
    for i in range(ns):
        for k in range(nxi):
                psi[:, i, k], psi_hat[:, i, k] = window_filter_1d(n, int(s[i]), xi[k])
    return psi, psi_hat

def gabor_wave_1d(n, s, xi):
    # generate one 1D gabor wavelet 
    x = np.arange(n) - np.floor(n/2)
    chi = np.zeros(n)
    chi[0:s] = 1/s
    
    o = np.exp(1j * xi * x)
    
    psi = np.multiply(chi, o)
    
    psi_hat = np.fft.fft(psi)
    return psi, psi_hat
                         

def gabor_wave_family_1d(n, s, xi):
    # generate a family of 1D gabor wavelets with specified scales and rotations in space
    ns = s.shape[0]
    nxi = xi.shape[0]
    
    psi = np.zeros((n, ns, nxi),dtype=complex)
    psi_hat = np.zeros((n, ns, nxi),dtype=complex)
    for i in range(ns):
        for k in range(nxi):
                psi[:, i, k], psi_hat[:, i, k] = gabor_wave_1d(n, int(s[i]), xi[k])
    return psi, psi_hat


def wave_trans_in_space_1d(x, psi):
    # wavelet transform in space
    n = psi.shape[1]
    f = np.zeros(psi.shape,dtype = complex)
    for i in range(n):
        f[:,i] = signal.convolve(x,psi[:,i],'same')
    return f


def wave_trans_in_freq_1d(x, psi_hat):
    # wavelet transform in frequency
    nx = x.shape
    
    x_hat = np.fft.fft(x)
    f = np.zeros(psi_hat.shape, dtype = complex)
    for i in range(psi_hat.shape[1]):
        f[:,i] = np.fft.ifft(np.multiply(x_hat, psi_hat[:,i]))
    return f

def diff(y0, sx, psi_hat, psi):
    # difference vector between first moment wavelet coefficients
    sy = np.sum(np.abs(wave_trans_in_freq_1d(y0, psi_hat)), axis = 0)
    sy = np.append(np.sum(y0), sy)
    
    diff = np.sum((sy - sx)**2)
    
    return diff

def synthesis(x, psi_hat, psi, jacob, max_err, max_epoch, *args):
    # collect parameters
    nx = x.shape
    nw = psi_hat.shape[1]  # number of wavelets 
    
    # scattering coefficients of original signal
    sx = np.sum(np.abs(wave_trans_in_freq_1d(x, psi_hat)),axis = 0)
    sx = np.append(np.sum(x), sx)
    
    # randomly initialize new signal
    narg = len(args)
    if narg == 0:
        y0 = np.random.random(nx[0]) 
    else:
        y0 = args[0]
        
    y = np.zeros((nx[0], 1))
    
    err = 1
    epoch = 0
    tic = time.time()
    while (err > max_err) & (epoch < max_epoch):
        
        epoch += 1
        ind = np.random.choice(nw, nw, replace = False) # randomize index of wavelets
        print('epoch:', epoch)
        
        for i in range(nw):
#             print('number of wavelets:', i + 1)
#             print('added index:', ind[i])
            if jacob:
                res = minimize(diff, y0, args = (sx[np.append([0], ind[0:i+1] + 1)], psi_hat[:,ind[0:i+1]], \
                                                         psi[:,ind[0:i+1]]), jac = jac, method='BFGS')
            else:
                res = minimize(diff, y0, args = (sx[np.append([0], ind[0:i+1] + 1)], psi_hat[:,ind[0:i+1]], \
                                                         psi[:,ind[0:i+1]]), method='BFGS')
            y0 = res.x
            y = np.append(y, np.reshape(y0, (y0.shape[0], 1)), axis = 1)
            
            err = res.fun
        print('current error:', err)
        
    toc = time.time()
    print('running time: ', toc - tic)
    return y

def jac(y0, sx, psi_hat, psi):
    # jacobian function for difference
    epsilon = 1e-6
    n = y0.shape[0]
    nw = psi_hat.shape[1]
    
    temp1 = wave_trans_in_freq_1d(y0, psi_hat)
    temp3 = np.zeros(n)
    
    psi_shift = np.zeros((n,n), dtype = complex)
    
    sy = np.sum(np.abs(temp1), axis = 0)
    sy = np.append(np.sum(y0), sy)
    temp3 = temp3 + 2 * (sy[0] - sx[0]) * np.ones(n)
    
    for i in range(nw):
        temp2 = temp1[:,i]
        for p in range(n):
            psi_shift[:, p] = np.roll(psi[:, i], p, axis = 0)
        
        temp = np.matmul(np.divide(np.real(temp2), abs(temp2) + epsilon), np.real(psi_shift)) + \
               np.matmul(np.divide(np.imag(temp2), abs(temp2) + epsilon), np.imag(psi_shift))
        temp3 = temp3 + 2 * (sy[i+1] - sx[i+1]) * temp
        
    return temp3

def jacfun(y0, sx, psi_hat, psi):
    # jacobian function for difference
    epsilon = 1e-6
    n = y0.shape[0]
    nw = psi_hat.shape[1]
    
    temp1 = wave_trans_in_freq_1d(y0, psi_hat)
    temp3 = np.zeros(n)
    
    psi_shift = np.zeros((n,n), dtype = complex)
    
    sy = np.sum(np.abs(temp1), axis = 0)
    
    for i in range(nw):
        temp2 = temp1[:,i]
        for p in range(n):
            psi_shift[:, p] = np.roll(psi[:, i], p, axis = 0)
        temp = np.matmul(np.divide(np.real(temp2), abs(temp2) + epsilon), np.real(psi_shift)) + \
               np.matmul(np.divide(np.imag(temp2), abs(temp2) + epsilon), np.imag(psi_shift))
        temp3 = temp3 + 2 * (sy[i] - sx[i]) * temp
    return temp3

def sample_poisson(t, l):
    # get a sample of poisson process
    
    #initialization
    x = np.zeros(1)
    t_sum = np.zeros(1)
    
    #sample time of jumps
    while t_sum[-1] < t[-1]:
        u = - np.log(np.random.uniform(0,1,1)) / l
        x = np.append(x, u)
        t_sum = np.append(t_sum, t_sum[-1] + u) 
    
    y = np.zeros(len(t))
    ind_left = 0
    
    #sample signal
    for i in range(len(x) - 1):
        ind_right = np.max(np.where(t < t_sum[i + 1])) + 1
        if ind_right > ind_left:
            y[ind_left:ind_right] = i 
        ind_left = ind_right
    
    return y, x

def prod_f(g_hat, t, nsample, l):
    # compute f_{\xi}(s) in the notes
    
    n = g_hat.shape[0]
    
    #generate samples from poisson process
    y = np.zeros((nsample, t.shape[0]))
    for i in range(nsample):
        y[i,:] = sample_poisson(t, l)[0]
    dy = np.diff(y, axis = 1) # compute difference
    
    #compute expectation of modulus of window filter transforms
    f = np.zeros((nsample, g_hat.shape[1]))
    for i in range(nsample):
        ind = np.random.choice(n,1)
        f[i, :] = np.abs(wave_trans_in_freq_1d(dy[i,:], g_hat)[ind, :])
    f = np.mean(f, axis = 0)
    
    return f

