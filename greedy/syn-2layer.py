import numpy as np
import math
import cmath
from scipy import signal
import random
import time

from fun1d import *
from poisson import *

from IPython import display
# from numbapro import vectorize

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
    
def diff(y0, sx, psi_hat, psi, m):
    # difference between scattering coefficients(up to 2 layers)
    nw = psi_hat.shape[1]*(psi_hat.shape[2])
    sy = scat_infreq_1d(y0, psi_hat, m)
    diff = 0
    for i in range(m + 1):
        if i < 2:
            # zero or 1st layer
            diff += np.sum((sx[i] - sy[i])**2)
        elif i == 2:
            # 2nd layer
            for j in range(nw):
                diff += np.sum((sx[i][j] - sy[i][j])**2)
#     print('diff:', diff)
    return diff

def syn_2layer(sx, psi_hat, psi, optim_pars, m, *args):
    #synthesis with scattering coefficients(up to 2 layers)
    
    # collect parameters
    max_err = optim_pars[0]
    max_epoch = optim_pars[1]
    n = psi_hat.shape[0]
    ns = psi_hat.shape[1]  # number of wavelets 
    nxi = psi_hat.shape[2]
    
    # randomly initialize new signal
    narg = len(args)
    if narg == 0:
        y0 = np.random.random(n) 
    else:
        y0 = args[0]
        
    y = np.zeros((n, 1))
    
    err = 1
    epoch = 0
    tic = time.time()
    
    while (epoch < max_epoch) & (max_err < err):
        
        epoch += 1
        ind = np.random.choice(ns, ns, replace = False) # randomize index of wavelets
        print('epoch:', epoch)
        
        # synthesize with only zero and 1st layer
        for i in range(ns):
            ind_s = ind[0:(i+1)]
            ind_w = np.array([])
            for j in range(nxi):
                ind_w = np.append(ind_w, ind_s * nxi + j)
            ind_w = np.sort(ind_w).astype(int)
            res = minimize(diff, y0, args = ([sx[0], sx[1][ind_w]], \
                                             psi_hat[:,ind_s,:], \
                                             psi[:,ind_s,:], \
                                             1), method='BFGS')
            y0 = res.x
            y = np.append(y, np.reshape(y0, (y0.shape[0], 1)), axis = 1)
            
            err = res.fun
            print('current error:', err)
        print('1st layer done!')
        if m > 1:
            # synthesis with 2nd layer added
            for i in range(ns):
                ind_s = ind[0:(i+1)]
                ind_w = np.array([])
                for j in range(nxi):
                    ind_w = np.append(ind_w, ind_s * nxi + j)
                ind_w = np.sort(ind_w).astype(int)
                # print('sx[2].shape:', sx[2][np.meshgrid(ind_w, ind_w)].shape)
                res = minimize(diff, y0, args = ([sx[0], sx[1][ind_w], sx[2][np.meshgrid(ind_w, ind_w)]], \
                                                 psi_hat[:,ind_s,:], \
                                                 psi[:,ind_s,:], \
                                                 2), jac = jac_2layer, method='BFGS')
                y0 = res.x
                y = np.append(y, np.reshape(y0, (y0.shape[0], 1)), axis = 1)
                err = res.fun
                print('current error:', err)
    toc = time.time()
    print('running time: ', toc - tic)
    return y

def jac_2layer(y0, sx, psi_hat, psi, m):
    # jacobian for minimizing scattering difference(up to 2 layers)
    # collect parameters
    epsilon = 1e-6
    n = y0.shape[0]
    ns = psi_hat.shape[1]
    nxi = psi_hat.shape[2]
    nw = ns * nxi
    
    temp1 = wave_trans_in_freq_1d(y0, np.reshape(psi_hat, (n, -1)))
    
    temp3 = np.zeros(n)
    psi_shift = np.zeros((n,n,nw), dtype = complex)
    psi_fftshift = np.fft.fftshift(np.reshape(psi, (n, -1)), axes = 0)
    
    # compute scattering for y0
    sy = scat_infreq_1d(y0, psi_hat, m)
    
    # add jacobian for zero order scattering
    temp3 = temp3 + 2 * (sy[0] - sx[0]) * np.ones(n)/n
    
    # shift wavelets
    for i in range(nw):
        for p in range(n):
            psi_shift[:, p, i] = np.roll(psi_fftshift[:,i], p, axis = 0)
            
    # add jacobian for first order scattering
    for i in range(nw):
        temp2 = temp1[:,i]
        temp = np.matmul(np.divide(np.real(temp2), abs(temp2) + epsilon), np.real(psi_shift[:,:,i])) + \
               np.matmul(np.divide(np.imag(temp2), abs(temp2) + epsilon), np.imag(psi_shift[:,:,i]))
        temp3 = temp3 + 2 * (sy[1][i] - sx[1][i]) * temp
        
    if m > 1:
        # add jacobian for second order scattering
        for i in range(nw - nxi):
            x_temp = abs(temp1[:, i])
            s_temp = i // nxi
            temp4 = wave_trans_in_freq_1d(x_temp, np.reshape(psi_hat[:, (s_temp+1):, :], (n, -1)))
            for l in range(temp4.shape[1]):
                temp5 = temp4[:, l]
                temp = np.matmul(np.divide(np.real(temp5), abs(temp5) + epsilon), \
                                 np.real(psi_shift[:,:,s_temp * nxi + l])) + \
                       np.matmul(np.divide(np.imag(temp5), abs(temp5) + epsilon), \
                                 np.imag(psi_shift[:,:,s_temp * nxi + l]))
                temp3 = temp3 + 2 * (sy[2][i, s_temp * nxi + l] - sx[2][i, s_temp * nxi + l]) * temp
    return temp3

# main
n = 2**10
m = 2
l = 0.01
pi = math.pi
max_err = 1e-7
max_epoch = 5
epsilon = 1e-4
optim_pars = [max_err, max_epoch]

# define filters
sigma = determine_sigma(epsilon)
Q = 2
alpha = 2
J = determine_J(n, Q, sigma, alpha)
s = np.unique(np.floor(2 ** np.linspace(0, J, int(J*Q)+1)))
xi = np.arange(1,3) * pi / 3
g, g_hat = window_filter_family_1d(n, s, xi, sigma)

# generate target
t = np.linspace(0, n + 1, n + 1)
mu = 1
x = sample_compound_poisson(t, l,'const',mu)
dx = np.diff(x)
sx = scat_infreq_1d(dx, g_hat, m)
# sx = prod_f(g_hat, t, nsample, l)

# synthesis
res = syn_2layer(sx, g_hat, g, optim_pars, m)
dx = np.reshape(dx, (dx.shape[0],1))
res = np.append(dx, res, axis = 1)
np.save('res_2layer1.npy', res)

