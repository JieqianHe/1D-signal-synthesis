Updated code for synthesis using GAN on hpcc. 

- pretrain discriminator with real data from poisson distribution with constant height and fake data from poisson with another \lambda with height from gaussian distribution.
- train generator:discriminator = 2:1
- train 300 epochs, but result get stuck before 100 epoch.
- Need to include data x0_data.npy and x0_gaussian_data.npy saved on pc: /Users/kejiqing/Desktop/research/synthesis/python/data. Newest result saved on pc: /Users/kejiqing/Desktop/research/synthesis/python/GAN-pt/result13
