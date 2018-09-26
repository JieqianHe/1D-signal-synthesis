import numpy as np
import math
import cmath
#import matplotlib.pyplot as plt
#from scipy import signal
#from scipy.optimize import least_squares
#from scipy.optimize import minimize
#from scipy.io import loadmat
import random
from numpy import linalg as LA
import time
# from numbapro import vectorize

def gaussian(x, sigma):
    pi = math.pi
    y = 1 / np.sqrt(2 * pi) * np.exp(- x**2 / 2)
    return y

def window_filter_1d(n, s, xi, sigma):
    # generate one 1D gabor wavelet 
    x = np.arange(n) - np.floor(n/2)
    chi = np.zeros(n)
    chi = gaussian(x/s, sigma)
    
#     o = np.exp(1j * xi * x)
    o = np.exp(1j * xi/s * x)
    
    psi = np.multiply(chi, o)
    
    psi_hat = np.fft.fft(np.fft.fftshift(psi))
#     psi_hat = np.fft.fft(psi)
    return psi, psi_hat

def window_filter_family_1d(n, s, xi, sigma):
    # generate a family of 1D gabor wavelets with specified scales and rotations in space
    ns = s.shape[0]
    nxi = xi.shape[0]
    
    psi = np.zeros((n, ns, nxi),dtype=complex)
    psi_hat = np.zeros((n, ns, nxi),dtype=complex)
    for i in range(ns):
        for k in range(nxi):
                psi[:, i, k], psi_hat[:, i, k] = window_filter_1d(n, int(s[i]), xi[k], sigma)
    return psi, psi_hat

def determine_sigma(epsilon):
    sigma = np.sqrt(- 2 * np.log(epsilon)) / math.pi
    return sigma

def determine_J(N, Q, sigma, *alpha):
    if len(alpha) == 0:
        alpha = 3
    J = np.log2(N) - np.log2(alpha) - np.log2(sigma) - 1
    int_J = max(np.floor(J), 1);
    frac_J = (1/Q) * np.around((J - int_J) * Q);
    J = int_J + frac_J;
    return J

def gabor_wave_1d(n, s, xi):
    # generate one 1D gabor wavelet 
    x = np.arange(n) - np.floor(n/2)
    chi = np.zeros(n)
    chi[0:s] = 1/s
    
    o = np.exp(1j * xi * x)
    
    psi = np.multiply(chi, o)
    
    psi_hat = np.fft.fft(psi)
    return psi, psi_hat
                         

def gabor_wave_family_1d(n, s, xi, sigma):
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

def scat_coeff(x, g_hat):
    f = np.sum(np.abs(wave_trans_in_freq_1d(x, g_hat)), axis = 0)
    f = np.append(np.sum(np.abs(x)), f)
    f = np.append(np.sum(x), f)
    return f


def scat_infreq_1d(x, psi_hat, m):
    # compute scattering coefficients to m order, currently for m = 1 or 2
    # collect parameters
    n = psi_hat.shape[0]
    ns = psi_hat.shape[1]
    nxi = psi_hat.shape[2]
    nw = ns * nxi
    
    # zeros order
    s0 = np.mean(x)
    
    #first order
    if m > 0:
        wx = np.abs(wave_trans_in_freq_1d(x, np.reshape(psi_hat,(n, -1))))
        s1 = np.mean(wx, axis = 0)
    if m == 1:
        return [s0, s1]
    elif m == 2:
        # second order
        s2 = np.zeros((nw, nw))
        for i in range(ns - 1):
            for j in range(nxi):
                temp = np.abs(wave_trans_in_freq_1d(wx[:,i*nxi + j], np.reshape(psi_hat[:,(i + 1):, :], (n, -1))))
                s2[i * nxi + j, (i+1)*nxi:] = np.mean(temp, axis = 0)
        return [s0, s1, s2]
