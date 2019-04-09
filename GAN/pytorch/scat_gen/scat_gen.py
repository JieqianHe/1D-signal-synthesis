from __future__ import print_function
from IPython import display
import torch
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.init as init
from torchvision import transforms, datasets
import argparse

import matplotlib
matplotlib.use('Agg')
import pylab as plt

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
#%matplotlib inline
from matplotlib import colors
from IPython import display
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly.graph_objs import Scatter, Figure, Layout
import numpy as np
import time
import math
import pytorch_fft.fft.autograd as fft
pi = math.pi

parser = argparse.ArgumentParser()
parser.add_argument('--test_id', type=int, required=True, help='id of current test')
parser.add_argument('--datasize', type=int, default=2**12, help='size of 1D signal')
parser.add_argument('--lambda_true', type=float, default=0.025, help='intensity of poisson from true data')
parser.add_argument('--nepochs', type=int, default=300, help='number of total epochs')
parser.add_argument('--batchsize', type=int, default=64, help='batchsize for each training step')
parser.add_argument('--layer2', type=bool, default=True, help='whether to do second layer scattering')
parser.add_argument('--l2', type=bool, default=False, help='whether to do l2 norm in scattering')
parser.add_argument('--norm', type=bool, default=False, help='whether to add scattering of normalized signal')
parser.add_argument('--ntestsample', type=int, default=16, help='number of samples for testing, >= 16')
parser.add_argument('--p1', type=int, default=1, help='scattering moments for the first layer')
parser.add_argument('--p2', type=int, default=1, help='scattering moments for the second layer')
parser.add_argument('--output_scale', type=float, default=5, help='maximal height of output signal from generator')
parser.add_argument('--Q', type=int, default=2, help='scale intervals for defining wavelets')
parser.add_argument('--J', type=int, default=4, help='largest scale for defining wavelets')
parser.add_argument('--xi', type=float, default=np.asarray([pi/6]), help='central frequency of 1st layer wavelets')
parser.add_argument('--xi2', type=float, default=np.asarray([pi/4]), help='central frequency of 2nd layer wavelets')
parser.add_argument('--c', type=int, default=2, help='s2/s1, proportion of wavelet scales between 2nd and 1st layer')
opt = parser.parse_args()

ntest=opt.test_id
n = opt.datasize
lambda_true = opt.lambda_true
num_epochs = opt.nepochs
batch_size = opt.batchsize
layer2 = opt.layer2
l2 = opt.l2
norm = opt.norm
num_test_samples = opt.ntestsample
output_scale = opt.output_scale
p1 = opt.p1
p2 = opt.p2
Q = opt.Q
J = opt.J
xi = opt.xi
xi2 = opt.xi2
c = opt.c
print('parameter defined!')

x_data = np.load('./data/x_homo_regular.npy')
ndata = x_data.shape[0]
x_data = torch.from_numpy(np.reshape(x_data,[-1,1,n])).float()
x_fake_data = np.load('./data/x_homo_fake.npy')
x_fake_data = torch.from_numpy(np.reshape(x_fake_data,[-1,1,n])).float()
nbatch = ndata // batch_size # number of batches in one epoch
print('Data loaded!')

class scattering(torch.nn.Module):
    def __init__(self, g_real_hat, g_imag_hat, g2_real_hat, g2_imag_hat, l2, layer2, norm, batch_size, n, p1, p2, eps = 1e-7):
        # Two-layer scattering module.
        
        # g_real_hat: real part of wavelets in frequency in the first scattering layer
        # g_imag_hat: imagery part of wavelets in frequency in the first scattering layer
        # g2_real_hat: real part of wavelets in frequency in the second scattering layer
        # g2_imag_hat: imagery part of wavelets in frequency in the second scattering layer
        # l2, layer2, norm: whether to do l2 norm, second layer scattering and signal normalization with heights {1,-1}
        # batch_size: length of batch
        # n: signal length
        # p1, p2: moments of first and second layer
        
        super(scattering, self).__init__()
        self.g_real_hat = g_real_hat.unsqueeze(0) # shape 1 * nwave * n
        self.g_imag_hat = g_imag_hat.unsqueeze(0)
        self.g2_real_hat = g2_real_hat.unsqueeze(0) # shape 1 * nwave * n
        self.g2_imag_hat = g2_imag_hat.unsqueeze(0)
        self.batch_size = batch_size
        self.nwave = g_real_hat.shape[0]
        self.x_imag = torch.zeros(self.batch_size, 1, n, 1) # imagery part of x is zero
        self.x2_imag = torch.zeros(self.batch_size, self.nwave, n, 1) 
        if torch.cuda.is_available():
            self.x_imag = self.x_imag.cuda()
            self.x2_imag = self.x2_imag.cuda()
        
        self.layer2 = layer2
        self.norm = norm
        self.p1 = p1
        self.p2 = p2
        self.eps = eps     
        
    def forward(self, x_real): 
        # x_real: batch_size * 1 * n
        self.x_real = x_real.unsqueeze(3)
        self.x = torch.cat((self.x_real, self.x_imag), 3) # batch_size * 1 * n * 2
        
        # convolution in frequency
        x_hat = torch.fft(self.x, 1) # fft
        # multiply in freq batch_size * nwave * n
        y_real_hat = x_hat[:,:,:,0] * self.g_real_hat - x_hat[:,:,:,1] * self.g_imag_hat 
        y_imag_hat = x_hat[:,:,:,0] * self.g_imag_hat + x_hat[:,:,:,1] * self.g_real_hat
        # ifft, batch_size * nwave * n * 2
        y = torch.ifft(torch.cat((y_real_hat.unsqueeze(3), y_imag_hat.unsqueeze(3)), 3), 1) 
        
        # nonlinear operator: modulus, batch_size * nwave * n
        temp = torch.sqrt(y[:,:,:,0]**2 + y[:,:,:,1]**2) 
        
        # 1st order scattering
        s = torch.mean(temp**self.p1, 2)
        
        # 2nd order scattering
        if self.layer2:
            temp = temp.unsqueeze(3)
            temp2 = torch.zeros(self.batch_size, nwave, n)
            if torch.cuda.is_available():
                temp2 = temp2.cuda()
            x2 = torch.cat((temp**self.p1, self.x2_imag), 3) # batch_size * nwave * n * 2
            x2_hat = torch.fft(x2, 1)
            for i in range(self.nwave):
                # batch_size * n
                y2_real_hat = x2_hat[:,i,:,0] * self.g2_real_hat[:,i,:] - x2_hat[:,i,:,1] * self.g2_imag_hat[:,i,:] 
                y2_imag_hat = x2_hat[:,i,:,0] * self.g2_imag_hat[:,i,:] + x2_hat[:,i,:,1] * self.g2_real_hat[:,i,:]
                # batch_size * n * 2
                y2 = torch.ifft(torch.cat((y2_real_hat.unsqueeze(2), y2_imag_hat.unsqueeze(2)), 2), 1) 
                
                # nonlinear operator: modulus, batch_size * nwave * n
                temp2[:, i, :] = torch.sqrt(y2[:,:,0]**2 + y2[:,:,1]**2) 
            s = torch.cat((s, torch.mean(temp2**self.p2, 2)), 1)
        
        # normalize signal to height {1, -1}
        if self.norm:
            self.z_imag = torch.zeros(self.batch_size, 1, n, 1) # imagery part of x is zero
            self.z2_imag = torch.zeros(self.batch_size, self.nwave, n, 1) 
            if torch.cuda.is_available():
                self.z_imag = self.z_imag.cuda()
                self.z2_imag = self.z2_imag.cuda()
                
            self.z_real = (torch.abs(self.x_real) > self.eps).float() * torch.sign(self.x_real)
            self.z = torch.cat((self.z_real, self.z_imag), 3) # batch_size * 1 * n * 2
        
            # convolution in frequency
            z_hat = torch.fft(self.z, 1) # fft
            # multiply in freq batch_size * nwave * n
            w_real_hat = z_hat[:,:,:,0] * self.g_real_hat - z_hat[:,:,:,1] * self.g_imag_hat 
            w_imag_hat = z_hat[:,:,:,0] * self.g_imag_hat + z_hat[:,:,:,1] * self.g_real_hat
            # ifft, n_batch * nwave * n * 2
            w = torch.ifft(torch.cat((w_real_hat.unsqueeze(3), w_imag_hat.unsqueeze(3)), 3), 1) 
            
            # nonlinear operator: modulus, batch_size * nwave * n
            temp = torch.sqrt(w[:,:,:,0]**2 + w[:,:,:,1]**2) 

            # 1st order scattering
            s = torch.cat((s, torch.mean(temp**self.p, 2)), 1)

            # 2nd order scattering
            if self.layer2:
                temp = temp.unsqueeze(3)
                temp2 = torch.zeros(self.batch_size, nwave, n)
                z2 = torch.cat((temp**self.p1, self.z2_imag), 3) # batch_size * nwave * n * 2
                z2_hat = torch.fft(z2, 1)
                for i in range(self.nwave):
                    # batch_size * n
                    w2_real_hat = z2_hat[:,i,:,0] * self.g2_real_hat[:,i,:] - z2_hat[:,i,:,1] * self.g2_imag_hat[:,i,:] 
                    w2_imag_hat = z2_hat[:,i,:,0] * self.g2_imag_hat[:,i,:] + z2_hat[:,i,:,1] * self.g2_real_hat[:,i,:]
                    
                    # batch_size * n * 2
                    w2 = torch.ifft(torch.cat((w2_real_hat.unsqueeze(2), w2_imag_hat.unsqueeze(2)), 2), 1) 
                    # nonlinear operator: modulus, batch_size * nwave * n
                    temp2[:, i, :] = torch.sqrt(w2[:,:,0]**2 + w2[:,:,1]**2) 
                s = torch.cat((s, torch.mean(temp2**self.p2, 2)), 1)
        return s.unsqueeze(1) # batch_size * 1 *  nf
        
              

class Generator(torch.nn.Module):
    
    def __init__(self, output_scale, out_active = 'tanh'):
        super(Generator, self).__init__()
        self.output_scale = output_scale

        self.l1 = nn.Sequential(
            nn.Linear(100, 1024),
            nn.Tanh()
        )
        self.l2 = nn.Sequential(
            nn.Linear(1024, 256*256),
            nn.Tanh()
        )
        
        self.conv1 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=256, out_channels=128, kernel_size=8,
                stride=4, padding=2, bias=False
            ),
            nn.BatchNorm1d(128),
            nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=128, out_channels=1, kernel_size=8,
                stride=4, padding=2, bias=False
            )
        )
        if out_active == 'tanh':
            self.out = torch.nn.Tanh()
        elif out_active == 'sigmoid':
            self.out = torch.nn.Sigmoid()

    def forward(self, x):
        # Project and reshape
        x = self.l1(x)
        x = self.l2(x)
        x = x.view(x.shape[0], 256, 16*16)
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        # Apply Tanh
        return self.output_scale * self.out(x)
    


# Noise
def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): return n.cuda()
    return n

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.00, 0.02)

def plot_signals(ntest, epoch, signals):
    n = 16
    fig = plt.figure(figsize = (10,10))
    for i in range(n):
        plt.subplot(4,4,i+1)
        plt.plot(signals[i,:])
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
    np.save('./result/scat_gen_signal%s_%s.npy'%(ntest, epoch), signals)
    plt.savefig('./result/scat_gen_signal%s_%s'%(ntest, epoch))

def gabor_wave_1d(n, s, xi):
    # generate one 1D gabor wavelet 
    x = np.arange(n)
    chi = np.zeros(n)
    chi[0:s] = 1/s
#     chi[0:s] = 1
    
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

def bump_wave_1d(n, s, xi):
    # generate one 1D gabor wavelet 
    x = np.arange(n)
    chi = np.zeros(n)
    t = np.arange(s - 1) + 1
    chi[1:s] = np.exp( - s**2 / (4 * t * s - 4 * t**2))
#     chi[0:s] = 1
    
    o = np.exp(1j * xi * x)
    
    psi = np.multiply(chi, o)
    
    psi_hat = np.fft.fft(psi)
    return psi, psi_hat

def bump_wave_family_1d(n, s, xi):
    # generate a family of 1D gabor wavelets with specified scales and rotations in space
    ns = s.shape[0]
    nxi = xi.shape[0]
    
    psi = np.zeros((n, ns, nxi),dtype=complex)
    psi_hat = np.zeros((n, ns, nxi),dtype=complex)
    for i in range(ns):
        for k in range(nxi):
                psi[:, i, k], psi_hat[:, i, k] = bump_wave_1d(n, int(s[i]), xi[k])
    return psi, psi_hat

def estimate_lambda(x, epsilon = 0.8):
    lambda_hat = np.zeros(x.shape[0])   
    for i in range(x.shape[0]):
        ind = np.append([0], np.where(x[i,:] > epsilon))
        if ind.shape[0] > 1:
            ind_delta = np.diff(ind).squeeze()
            lambda_hat[i] = 1/np.mean(ind_delta)
    return np.mean(lambda_hat)

# define wavelets

# epsilon = 1e-4
# sigma = determine_sigma(epsilon)
# Q = 2
# alpha = 2
# J = determine_J(n, Q, sigma, alpha)
# J = 4
s = np.unique(np.floor(2 ** np.linspace(1, J, int(J*Q)+1-Q)))
#s = np.array([1,2,4,8,10,16,31,63,129,257,513])
# xi = 2 * pi * np.random.choice(n,1) / n
# xi2 = 2 * pi * np.random.choice(n,1) / n
# np.save('./result/xi_%s.npy'%ntest, xi)
# np.save('./result/xi2_%s.npy'%ntest, xi2)

g, g_hat = gabor_wave_family_1d(n,s,xi)
g = np.reshape(g, (g.shape[0], -1)) # n * nwave
nwave = g.shape[1]
g_hat = np.swapaxes(np.reshape(g_hat, (g_hat.shape[0], -1)), 0, 1) # nwave * n
g_real_hat = torch.from_numpy(np.real(g_hat)).float()
g_imag_hat = torch.from_numpy(np.imag(g_hat)).float()

s2 = s * c
g2, g2_hat = gabor_wave_family_1d(n,s2,xi2)
g2 = np.reshape(g2, (g2.shape[0], -1)) # n * nwave
g2_hat = np.swapaxes(np.reshape(g2_hat, (g2_hat.shape[0], -1)), 0, 1) # nwave * n
g2_real_hat = torch.from_numpy(np.real(g2_hat)).float()
g2_imag_hat = torch.from_numpy(np.imag(g2_hat)).float()

if torch.cuda.is_available():
    g_real_hat = g_real_hat.cuda()
    g_imag_hat = g_imag_hat.cuda()
    g2_real_hat = g2_real_hat.cuda()
    g2_imag_hat = g2_imag_hat.cuda()
print('wavelets defined')



test_noise = noise(num_test_samples)
# logger = Logger(model_name='DCGAN', data_name='MNIST')
g_error_sum = []


# Create Network instances and init weights
generator = Generator(output_scale)
generator.apply(init_weights)
# generator.load_state_dict(torch.load('./result/scat_GAN_GEN_test%s'%(ntest - 1), map_location=lambda storage, loc: storage))

nf = nwave
if l2:
    nf = 2 * nf
if layer2:
    nf = 2 * nf
if norm:
    nf = 2 * nf
print('nf: ',nf)   

scatter = scattering(g_real_hat, g_imag_hat, g2_real_hat, g2_imag_hat, l2, layer2, norm, batch_size, n, p1, p2)
scatter.apply(init_weights)

# Enable cuda if available
if torch.cuda.is_available():
    generator.cuda()
    scatter.cuda()
    
# Optimizers
g_optimizer = Adam(generator.parameters(), lr=0.000005, betas=(0.5, 0.999))

# Loss function
loss = nn.MSELoss()

print('Model initialized!')

for epoch in range(num_epochs):
    print('epoch:',epoch)
    for idx in range(nbatch):
        real_batch = x_data[idx * batch_size:(idx + 1)*batch_size, :]
        # 1. Train Discriminator
        real_data = Variable(real_batch)
        if torch.cuda.is_available(): real_data = real_data.cuda()
        s_real = scatter(real_data)
        
        # Train Generator
        # Generate fake data
        fake_data = generator(noise(real_batch.size(0)))
        # Train G
        # g_error = train_generator(g_optimizer, fake_data)
        g_optimizer.zero_grad()
        # Sample noise and generate fake data
        s_fake = scatter(fake_data)
        # Calculate error and backpropagate
        #print('prediction fake: ', prediction[0])
        g_error = loss(s_fake, s_real)
        g_error.backward()
        # Update weights with gradients
        g_optimizer.step()
        # Return error
        
        #print(torch.mean(d_error).data.cpu())
        g_error_sum.append(float(torch.mean(g_error).data.cpu()))
        
    test_signals = generator(test_noise).squeeze(1).data.cpu().numpy()
    plot_signals(ntest, epoch, test_signals)
    lamb_est = estimate_lambda(test_signals)
    print('estimated lambda: ', lamb_est)
    np.save('./result/scat_gen_loss%s.npy'%ntest, np.asarray(g_error_sum))
    torch.save(generator.state_dict(), './result/scat_gen_G%s'%ntest)
