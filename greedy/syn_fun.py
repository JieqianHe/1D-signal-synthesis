import numpy as np
import math
import cmath
from scipy import signal
import random
import time
from fun1d import *

def diff(y0, sx, psi_hat, psi, z):
    # difference vector between first moment wavelet coefficients
    sy = scat_coeff(y0, psi_hat)
    if z:
        diff = np.sum((sy - sx)**2)
    else:
        diff = np.sum((sy[2:] - sx[2:])**2)
    return diff

def synthesis(sx, psi_hat, psi, jacob, max_err, max_epoch, z, *args):
    # collect parameters
    nx = psi_hat.shape[0]
    nw = psi_hat.shape[1]  # number of wavelets 

    # randomly initialize new signal
    narg = len(args)
    if narg == 0:
        y0 = np.random.random(nx)
    else:
        y0 = args[0]

    y = np.zeros((nx, 1))

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
                res = minimize(diff, y0, args = (sx[np.append([0, 1], ind[0:i+1] + 2)], psi_hat[:,ind[0:i+1]], \
                                                         psi[:,ind[0:i+1]], z), jac = jac, method='BFGS')
            else:
                res = minimize(diff, y0, args = (sx[np.append([0, 1], ind[0:i+1] + 2)], psi_hat[:,ind[0:i+1]], \
                                                         psi[:,ind[0:i+1]], z), method='BFGS')
            y0 = res.x
            y = np.append(y, np.reshape(y0, (y0.shape[0], 1)), axis = 1)

            err = res.fun
            print('current error:', err)

    toc = time.time()
    print('running time: ', toc - tic)
    return y

def jac(y0, sx, psi_hat, psi, z):
    # jacobian function for difference
    epsilon = 1e-6
    n = y0.shape[0]
    nw = psi_hat.shape[1]

    temp1 = wave_trans_in_freq_1d(y0, psi_hat)
    temp3 = np.zeros(n)
    psi_shift = np.zeros((n,n), dtype = complex)
    psi_fftshift = np.fft.fftshift(psi, axes = 0)

    sy = scat_coeff(y0, psi_hat)
    if z:
        temp3 = temp3 + 2 * (sy[0] - sx[0]) * np.ones(n)
        temp3 = temp3 + 2 * (sy[1] - sx[1]) * np.sign(y0)

    for i in range(nw):
        temp2 = temp1[:,i]
        for p in range(n):
            psi_shift[:, p] = np.roll(psi_fftshift[:,i], p, axis = 0)
        temp = np.matmul(np.divide(np.real(temp2), abs(temp2) + epsilon), np.real(psi_shift)) + \
               np.matmul(np.divide(np.imag(temp2), abs(temp2) + epsilon), np.imag(psi_shift))
        temp3 = temp3 + 2 * (sy[i+2] - sx[i+2]) * temp
    return temp3
