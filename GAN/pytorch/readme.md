Updated code for generating signals from poisson process using GAN in pytorch. 

*syn-gan-pt.py*:
- pretrain discriminator with real data from poisson distribution with constant height and fake data from poisson with another &lambda with height from gaussian distribution.
- train generator:discriminator = 2:1
- train 300 epochs, but result get stuck before 100 epoch.
- runpytorch.qsub for submit job on hpcc

- Need to include data *x0_data.npy* and *x0_gaussian_data.npy* saved on pc:
  /Users/kejiqing/Desktop/research/synthesis/python/data. 
  Newest result saved on pc: 
  /Users/kejiqing/Desktop/research/synthesis/python/GAN-pt/result13
- Need to figure out how to save trained model and load model with sequence used in defining architecture.

*batch_disc_GAN.py*:
- implement minibatch discrimination with smooth target(0.1 vs. 0.9)
- solved problem with generating one single point(same fake signal, no randomness)
- estimated $\lambda$ is higher than real $\lambda$

*scat_GAN.py*:
- implement scattering GAN as $D(S(G(z)))$ with batch discrimination
- need to fix 'NaN' problem

*scat_GAN_isnan.py*:
- check 'nan' by using $torch.isnan$ and print out 'nan' values

*scat_GAN.sb*:
- sbatch file to run *scat_GAN.py* on hpcc

*result_scat.ipynb*:
- check result from scattering GAN, including load model and visualiza generated signals
