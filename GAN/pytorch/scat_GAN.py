from __future__ import print_function
from IPython import display
import torch
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.init as init
from torchvision import transforms, datasets

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

x_data = np.load('./data/x4_4096_data.npy')
ndata = x_data.shape[0]
x_data = torch.from_numpy(np.reshape(x_data,[-1,1,2**12])).float()
x_fake_data = np.load('./data/x4_4096_fake_data.npy')
x_fake_data = torch.from_numpy(np.reshape(x_fake_data,[-1,1,2**12])).float()
print('Data loaded!')

class scattering(torch.nn.Module):
    def __init__(self, g_real_hat, g_imag_hat, nbatch, n):
        # g_real_hat: real part of wavelets in frequency
        # g_imag_hat: imagery part of wavelets in frequency
        # nbatch: length of batch
        # n: signal length
        super(scattering, self).__init__()
        self.g_real_hat = g_real_hat.unsqueeze(0) # shape 1 * nwave * n
        self.g_imag_hat = g_imag_hat.unsqueeze(0)
        self.x_imag = torch.zeros(nbatch, 1, n, 1) # imagery part of x is zero
        if torch.cuda.is_available():
            self.x_imag = self.x_imag.cuda()
    def forward(self, x_real): 
        # x_real: n_batch * 1 * n
        self.x_real = x_real.unsqueeze(3)
        self.x = torch.cat((self.x_real, self.x_imag), 3) # n_batch * 1 * n * 2
        
        # convolution in frequency
        x_hat = torch.fft(self.x, 1) # fft
        y_real_hat = x_hat[:,:,:,0] * self.g_real_hat - x_hat[:,:,:,1] * self.g_imag_hat # multiply in freq n_batch * nwave * n
        y_imag_hat = x_hat[:,:,:,0] * self.g_imag_hat + x_hat[:,:,:,1] * self.g_real_hat # multiply in freq
        y = torch.ifft(torch.cat((y_real_hat.unsqueeze(3), y_imag_hat.unsqueeze(3)), 3), 1) # ifft, n_batch * nwave * n * 2
        temp = torch.sqrt(y[:,:,:,0]**2 + y[:,:,:,1]**2) # nonlinear operator: modulus, n_batch * nwave * n
        
        # scattering
        s = torch.sum(x_real, 2)
        s = torch.cat((s, torch.sum(temp, 2)), 1) # integration
        return s.unsqueeze(1)
        
              
class Discriminator(torch.nn.Module):
    
    def __init__(self, batch_disc, ns, n):
        super(Discriminator, self).__init__()
        self.ns = ns # number of scattering coefficients
        self.n = n # signal length
        self.in_features = 1024
        self.out_features = 128
        self.kernel_dims = 16
        self.mean = False
        
        self.l0 = nn.Linear(self.ns, self.n)
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=128, kernel_size=16, 
                stride=4, padding=6, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=128, out_channels=256, kernel_size=16,
                stride=4, padding=6, bias=False
            ),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.l1 = nn.Sequential(
            nn.Linear(256*256, self.in_features),
            nn.LeakyReLU(0.2, inplace=True)
        ) 
        
#        self.T = nn.Parameter(torch.Tensor(self.in_features, self.out_features, self.kernel_dims))
#        init.normal_(self.T, 0, 1)
        self.batch_disc = batch_disc
        if self.batch_disc:
            self.T = nn.Parameter(torch.randn(self.in_features, self.out_features, self.kernel_dims)*0.1, requires_grad = True)
            self.out = nn.Sequential(
                nn.Linear(self.in_features + self.out_features, 1),
                nn.Sigmoid(),
            )
        else:
            self.out = nn.Sequential(
                nn.Linear(self.in_features, 1),
                nn.Sigmoid(),
            )

    def forward(self, x):
        x = self.l0(x)
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        # Flatten and apply sigmoid
        x = x.view(-1, 256*16*16)
        x = self.l1(x)
        
        if self.batch_disc:
            # minibatch discrimination after third layer(fully connected)
            matrices = x.mm(self.T.view(self.in_features, -1))
            matrices = matrices.view(-1, self.out_features, self.kernel_dims)

            M = matrices.unsqueeze(0)  # 1xNxBxC
            M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
            norm = torch.abs(M - M_T).sum(3)  # NxNxB
            expnorm = torch.exp(-norm)
            o_b = (expnorm.sum(0) - 1)   # NxB, subtract self distance
            if self.mean:
                o_b /= x.size(0) - 1

            x = torch.cat([x, o_b], 1) # concatenate l1 feature and cross sample feature
        
        x = self.out(x)
        return x
class Generator(torch.nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        
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
        return self.out(x)
    


# Noise
def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): return n.cuda()
    return n

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.00, 0.02)

def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1) * 0.9)
    if torch.cuda.is_available(): return data.cuda()
    return data

def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.ones(size, 1) * 0.1)
    if torch.cuda.is_available(): return data.cuda()
    return data

def train_discriminator(optimizer, real_data, fake_data):
    # Reset gradients
    optimizer.zero_grad()
    
    # 1. Train on Real Data
    prediction_real = discriminator(scatter(real_data))
    # Calculate error and backpropagate
    # assert (prediction_real >= 0. & prediction_real <= 1.).all()
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    # 2. Train on Fake Data
    prediction_fake = discriminator(scatter(fake_data))
    # Calculate error and backpropagate
    # assert (prediction_fake >= 0. & prediction_fake <= 1.).all()
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()
    
    # Update weights with gradients
    optimizer.step()
    
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data):
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(scatter(fake_data))
    # Calculate error and backpropagate
    error = loss(prediction, real_data_target(prediction.size(0)))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error

def plot_signals(ntest, epoch, signals):
    n = 16
    fig = plt.figure(figsize = (10,10))
    for i in range(n):
        plt.subplot(4,4,i+1)
        plt.plot(signals[i,:])
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
    plt.savefig('./result/syn_pois_test%s_epoch%s'%(ntest, epoch))

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

def estimate_lambda(x, epsilon = 0.8):
    lambda_hat = np.zeros(x.shape[0])   
    for i in range(x.shape[0]):
        ind = np.append([0], np.where(x[i,:] > epsilon))
        if ind.shape[0] > 1:
            ind_delta = np.diff(ind).squeeze()
            lambda_hat[i] = 1/np.mean(ind_delta)
    return np.mean(lambda_hat)

# define wavelets
n = 2**12
pi = math.pi
epsilon = 1e-4
sigma = determine_sigma(epsilon)
Q = 2
alpha = 2
J = determine_J(n, Q, sigma, alpha)
s = np.unique(np.floor(2 ** np.linspace(0, J, int(J*Q)+1)))
#s = np.array([1,2,4,8,10,16,31,63,129,257,513])
xi = 2 * pi * np.random.choice(n,2) / n
g, g_hat = gabor_wave_family_1d(n,s,xi)
g = np.reshape(g, (g.shape[0], -1)) # n * nwave
nwave = g.shape[1]
g_hat = np.swapaxes(np.reshape(g_hat, (g_hat.shape[0], -1)), 0, 1) # nwave * n
g_real_hat = torch.from_numpy(np.real(g_hat)).float()
g_imag_hat = torch.from_numpy(np.imag(g_hat)).float()
if torch.cuda.is_available():
    g_real_hat = g_real_hat.cuda()
    g_imag_hat = g_imag_hat.cuda()
print('wavelets defined')

# Number of epochs
n = 2**12
lambda_true = 0.025
num_epochs = 200
pretrain_epochs = 10
batch_size = 64
nbatch = ndata // batch_size
num_test_samples = 16
batch_disc = True # do minibatch discrimination
test_noise = noise(num_test_samples)
# logger = Logger(model_name='DCGAN', data_name='MNIST')
d_error_sum = []
g_error_sum = []
fake_score_sum = []
real_score_sum = []
ntest = 0

# Create Network instances and init weights
generator = Generator()
generator.apply(init_weights)

discriminator = Discriminator(batch_disc, nwave + 1, n)
discriminator.apply(init_weights)

scatter = scattering(g_real_hat, g_imag_hat, batch_size, n)
scatter.apply(init_weights)

# Enable cuda if available
if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
    scatter.cuda()
    
# Optimizers
d_optimizer = Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
loss = nn.BCELoss()

print('Model initialized!')

for epoch in range(pretrain_epochs):
    for idx in range(nbatch):
        print('idx:',idx)
        real_batch = x_data[idx * batch_size:(idx + 1)*batch_size, :]
        real_data = Variable(real_batch)
        
        fake_batch = x_fake_data[idx * batch_size:(idx + 1)*batch_size, :]
        fake_data = Variable(fake_batch)
        if torch.cuda.is_available(): 
            real_data = real_data.cuda()
            fake_data = fake_data.cuda()
        # Train D
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, 
                                                                real_data, fake_data)


for epoch in range(num_epochs):
    print('epoch:',epoch)
    for idx in range(nbatch):
        real_batch = x_data[idx * batch_size:(idx + 1)*batch_size, :]
        # 1. Train Discriminator
        real_data = Variable(real_batch)
        if torch.cuda.is_available(): real_data = real_data.cuda()
        # Generate fake data
        fake_data = generator(noise(real_data.size(0))).detach()
        # Train D
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, 
                                                                real_data, fake_data)

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(real_batch.size(0)))
        # Train G
        g_error = train_generator(g_optimizer, fake_data)
        
        fake_data = generator(noise(real_batch.size(0)))
        g_error = train_generator(g_optimizer, fake_data)
    
        #print(torch.mean(d_error).data.cpu())
        d_error_sum.append(float(torch.mean(d_error).data.cpu()))
        g_error_sum.append(float(torch.mean(g_error).data.cpu()))
        fake_score_sum.append(float(torch.mean(d_pred_fake).data.cpu()))
        real_score_sum.append(float(torch.mean(d_pred_real).data.cpu()))
        
    test_signals = generator(test_noise).squeeze(1).data.cpu().numpy()
    plot_signals(ntest, epoch, test_signals)
    lamb_est = estimate_lambda(test_signals)
    print('estimated lambda: ', lamb_est)
    if np.abs(lamb_est - lambda_true) < 0.005:
            torch.save(discriminator.state_dict(), './result/scat_GAN_DISC_test%s_epoch%s'%(ntest, epoch))
            torch.save(generator.state_dict(), './result/scat_GAN_GEN_test%s_epoch%s'%(ntest, epoch))
    np.save('./result/scat_score_real_%s.npy'%ntest, np.asarray(real_score_sum))
    np.save('./result/scat_score_fake_%s.npy'%ntest,  np.asarray(fake_score_sum))
    np.save('./result/scat_loss_gen_%s.npy'%ntest, np.asarray(g_error_sum))
    np.save('./result/scat_loss_dis_%s.npy'%ntest, np.asarray(d_error_sum))
    torch.save(discriminator.state_dict(), './result/scat_GAN_DISC_test%s'%ntest)
    torch.save(generator.state_dict(), './result/scat_GAN_GEN_test%s'%ntest)
