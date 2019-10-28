
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distrib
import torch.distributions.transforms as transform
import matplotlib.animation as animation
# Imports for plotting
import numpy as np
import pickle
from torchvision import datasets
from torchvision import transforms
from Flow import *
from vae import *
import argparse
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description="VAE with Normalizing Flow")
parser.add_argument("--batch_size",type=int,
                    default=1,
                    help="""batch size for training""")
parser.add_argument("--epoch",type=int,
                    default=0,
                    help="""which epoch weights to use""")
parser.add_argument("--n_lambda",type=int,
                    default=3,
                    help="""number of intervals in the flow""")
parser.add_argument("--learning_rate",type=float,
                    default=0.001,
                    help="""learning rate""")
parser.add_argument("--n_epochs",type=int,
                    default=1000,
                    help="""number of training epochs""")
parser.add_argument("--n_subsamples",type=int,
                    default=500,
                    help="""number of samples used in training""")

args = parser.parse_args()

batch_size =args.batch_size
learning_rate=args.learning_rate

#transform the MNIST dataset to pytorch tensor
tens_t = transforms.ToTensor()
train_dset = datasets.MNIST('./data', train=True, download=True, transform=tens_t)
train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True)
test_dset = datasets.MNIST('./data', train=False, transform=tens_t)
test_loader = torch.utils.data.DataLoader(test_dset, batch_size=batch_size, shuffle=True)
#here we only use a subset of the whole dataset
subsample=args.n_subsamples
epochs=args.n_epochs
n_lambda=args.n_lambda
#encoder of particle flow
encoder=construct_encoder_particle_flow()
decoder=construct_decoder()
parameters=[]
for _,parameter in enumerate(encoder.parameters()):
    parameters.append(parameter)
for _,parameter in enumerate(decoder.parameters()):
    parameters.append(parameter)
parameters=nn.ParameterList(parameters)
optimizer = optim.Adam(parameters, lr=learning_rate)
losses = torch.zeros(epochs, 3)
for it in range(epochs):
    n_batch=0
    for batch_idx,(x,_) in enumerate(train_loader):
        if (batch_idx*batch_size)>subsample:
            break
        mu,sigma=encoder(x)
        q = distrib.Normal(torch.zeros(mu.shape[1]), torch.ones(sigma.shape[1]))
        z_0 = ((sigma) * q.sample((1,))) + mu
        flow = particle_flow(decoder,mu,sigma,x,n_lambda=n_lambda)
        z_k=flow(z_0)
        x_tilde=decoder(z_k)
        log_p_zk = (-0.5 * z_k * z_k).sum()
        log_q_z0 = (-sigma.log() -0.5* (z_0 - mu) * (z_0 - mu) /(sigma**2)).sum()
        log_q_z0=log_q_z0+flow.gamma
        #losses of latent variables z_0 and z_k
        loss_latent=log_q_z0-log_p_zk
        #reconstruction error
        loss_recons = reconstruction_loss(x_tilde, x, num_classes=1)
        loss = loss_recons + ( loss_latent)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch:',it+1,'iter:',batch_idx,'loss:',loss.item(),'reconstruction_loss:',loss_recons,'log_q_z0:',log_q_z0,'log_p_zk:',log_p_zk)
        losses[it,0]+=loss_recons.item()
        losses[it,1]+=loss_latent.item()
        losses[it, 2] += loss_recons.item() + loss_latent.item()
        n_batch+=1
    losses[it,:]/=n_batch
    print(("Epoch:{:>4}, loss:{:>4.2f}").format(it + 1, losses[it, 0].item() + losses[it, 1].item()))

plt.plot(range(epochs), (losses[:,2]).detach().numpy())
plt.show()

np.save('particle_flow_'+str(n_lambda)+'_intervals.npy',losses)
torch.save(encoder.state_dict(),
                   ("encoder_particle_flow.model"))
torch.save(decoder.state_dict(),
                   ("decoder_particle_flow.model"))


