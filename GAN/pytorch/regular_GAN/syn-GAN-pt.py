from __future__ import print_function
from IPython import display
import torch
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable

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

from poisson import *
from fun1d import *

x_data = np.load('x0_data.npy')
ndata = x_data.shape[0]
x_data = torch.from_numpy(np.reshape(x_data,[-1,1,2**12])).float()
x_fake_data = np.load('x0_gaussian_data.npy')
x_fake_data = torch.from_numpy(np.reshape(x_fake_data,[-1,1,2**12])).float()
print('Data loaded!')

class DiscriminativeNet(torch.nn.Module):
    
    def __init__(self):
        super(DiscriminativeNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=128, kernel_size=8, 
                stride=4, padding=2, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=128, out_channels=256, kernel_size=8,
                stride=4, padding=2, bias=False
            ),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=256, out_channels=512, kernel_size=8,
                stride=4, padding=2, bias=False
            ),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(
                in_channels=512, out_channels=1024, kernel_size=8,
                stride=4, padding=2, bias=False
            ),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Sequential(
            nn.Linear(1024*16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Flatten and apply sigmoid
        x = x.view(-1, 1024*16)
        x = self.out(x)
        return x

class GenerativeNet(torch.nn.Module):
    
    def __init__(self):
        super(GenerativeNet, self).__init__()
        
        self.linear = torch.nn.Linear(100, 1024*16)
        
        self.conv1 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=1024, out_channels=512, kernel_size=16,
                stride=4, padding=6, bias=False
            ),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=512, out_channels=256, kernel_size=16,
                stride=4, padding=6, bias=False
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=256, out_channels=128, kernel_size=16,
                stride=4, padding=6, bias=False
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=128, out_channels=1, kernel_size=16,
                stride=4, padding=6, bias=False
            )
        )
        #self.out = torch.nn.Tanh()
        self.out = torch.nn.ReLU()

    def forward(self, x):
        # Project and reshape
        x = self.linear(x)
        x = x.view(x.shape[0], 1024, 16)
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Apply Tanh
        return self.out(x)

class Discriminator(torch.nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
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
            nn.Linear(256*256, 1024),
            nn.LeakyReLU(0.2, inplace=True)
        ) 
        self.out = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        # Flatten and apply sigmoid
        x = x.view(-1, 256*16*16)
        x = self.l1(x)
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
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

def train_discriminator(optimizer, real_data, fake_data):
    # Reset gradients
    optimizer.zero_grad()
    
    # 1. Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    # 2. Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()
    
    # Update weights with gradients
    optimizer.step()
    
    return error_real + error_fake, prediction_real, prediction_fake
    return (0, 0, 0)

def train_generator(optimizer, fake_data):
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, real_data_target(prediction.size(0)))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error

def plot_signals(ntest, epoch, n_batch, signals):
    n = 16
    fig = plt.figure(figsize = (10,10))
    for i in range(n):
        plt.subplot(4,4,i+1)
        plt.plot(signals[i,:])
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
    plt.savefig('./result_tanh/syn_pois_test%s_epoch%s_%s'%(ntest, epoch, n_batch // 100))
    
# Create Network instances and init weights
generator = Generator()
generator.apply(init_weights)

discriminator = Discriminator()
discriminator.apply(init_weights)

# Enable cuda if available
if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
    
# Optimizers
d_optimizer = Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
loss = nn.BCELoss()

# Number of epochs
num_epochs = 300
pretrain_epochs = 10
batch_size = 100
nbatch = ndata // batch_size
num_test_samples = 16
test_noise = noise(num_test_samples)
# logger = Logger(model_name='DCGAN', data_name='MNIST')
d_error_sum = []
g_error_sum = []
fake_score_sum = []
real_score_sum = []
ntest = 13
print('Model initialized!')

for epoch in range(pretrain_epochs):
    for idx in range(nbatch):
        #print('idx:',idx)
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
    for idx in range(nbatch):
        #print('idx:',idx)
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
        
        if torch.mean(d_pred_fake) > torch.mean(d_pred_real):
            display.clear_output(True)
            # Display Images
            test_signals = generator(test_noise).squeeze(1).data.cpu().numpy()
            np.save('./result_tanh/fake_signal_test%s_epoch%s_idx%s.npy'%(ntest, epoch, idx), test_signals)
            plot_signals(ntest, epoch, idx, test_signals)
        # Display Progress
        if (idx % 100 == 99):
            display.clear_output(True)
            # Display Images
            test_signals = generator(test_noise).squeeze(1).data.cpu().numpy()
            np.save('./result_tanh/fake_signal_test%s_epoch%s.npy'%(ntest, epoch), test_signals)
            plot_signals(ntest, epoch, idx, test_signals)
            
#             plot_distributions(n_batch, test_images, d_error_sum, real_score_sum, 
#                                fake_score_sum, g_error_sum)
            np.save('./result_tanh/pois0_score_real_%s.npy'%ntest, np.asarray(real_score_sum))
            np.save('./result_tanh/pois0_score_fake_%s.npy'%ntest,  np.asarray(fake_score_sum))
            np.save('./result_tanh/pois0_loss_gen_%s.npy'%ntest, np.asarray(g_error_sum))
            np.save('./result_tanh/pois0_loss_dis_%s.npy'%ntest, np.asarray(d_error_sum))
            #torch.save(discriminator.state_dict(), './result_tanh/pois0_GAN_MINST_DISC_test%s_epoch%s'%(ntest, epoch))
            #torch.save(generator.state_dict(), './result_tanh/pois0_GAN_MINST_GEN_test%s_epoch%s'%(ntest,epoch))
