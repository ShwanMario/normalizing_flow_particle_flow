import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distrib
import torch.distributions.transforms as transform
from torch.utils.data import Dataset, DataLoader
# Imports for plotting
import numpy as np
import matplotlib.pyplot as plt
from Flow import *
# Define grids of points (for later plots)


class MNIST_Dataset(Dataset):
    def __init__(self, image):
        super(MNIST_Dataset).__init__()
        self.image = image
    def __len__(self):
        return self.image.shape[0]
    def __getitem__(self, idx):
        return np.random.binomial(1, self.image[idx, :]).astype('float32')


class VAE(nn.Module):

    def __init__(self, encoder, decoder, encoder_dims=450, latent_dims=64):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dims = latent_dims
        self.encoder_dims = encoder_dims
        self.mu = nn.Linear(encoder_dims, latent_dims)
        self.sigma = nn.Sequential(
            nn.Linear(encoder_dims, latent_dims),
            nn.Softplus())
        self.apply(self.init_parameters)

    def init_parameters(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # Encode the inputs
        z_params = self.encode(x)
        # Obtain latent samples and latent loss
        z_tilde, kl_div = self.latent(x, z_params)
        # Decode the samples
        x_tilde = self.decode(z_tilde)
        return x_tilde, kl_div

    def latent(self, x, z_params):
        n_batch = x.size(0)
        # Retrieve mean and var
        mu, sigma = z_params
        # Re-parametrize
        q = distrib.Normal(torch.zeros(mu.shape[1]), torch.ones(sigma.shape[1]))
        z = (sigma * q.sample((n_batch,))) + mu
        # Compute KL divergence
        kl_div = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        kl_div = kl_div / n_batch
        return z, kl_div

class construct_encoder(nn.Module):
    def __init__(self, latent_size=64,n_hidden=450,kernel_size=5,padding=2,channel=(16,32,32),stride=2,activation=nn.Softplus(),view_size=512,input_channel=1):
        super(construct_encoder, self).__init__()
        self.latent_size=latent_size
        self.n_hidden=n_hidden
        self.activation=activation
        self.channel=channel
        self.stride=stride
        self.padding=padding
        self.kernel_size=kernel_size
        self.conv1=nn.Conv2d(input_channel, self.channel[0], self.kernel_size, stride=self.stride, padding=self.padding)
        self.conv2=nn.Conv2d(self.channel[0], self.channel[1], self.kernel_size, stride=self.stride, padding=self.padding)
        self.conv3=nn.Conv2d(self.channel[1], self.channel[2], self.kernel_size, stride=self.stride, padding=self.padding)
        self.fc1=nn.Linear(view_size, self.n_hidden)

    def forward(self,inputs):
        output = self.activation(self.conv1(inputs))
        output = self.activation(self.conv2(output))
        output = self.activation(self.conv3(output))
        output = output.view(-1,output.shape[1]*output.shape[2]*output.shape[3])
        output = self.fc1(output)
        output = self.activation(output)
        return output

#the structure is the same as the network structure used in normalizing flow
class construct_encoder_particle_flow(nn.Module):
    def __init__(self, latent_size=64,n_hidden=450,kernel_size=5,padding=2,channel=(16,32,32),stride=2,activation=nn.Softplus(),view_size=512,input_channel=1):
        super(construct_encoder_particle_flow, self).__init__()
        self.latent_size=latent_size
        self.n_hidden=n_hidden
        self.activation=activation
        self.channel=channel
        self.stride=stride
        self.padding=padding
        self.kernel_size=kernel_size
        self.conv1=nn.Conv2d(input_channel, self.channel[0], self.kernel_size, stride=self.stride, padding=self.padding)
        self.conv2=nn.Conv2d(self.channel[0], self.channel[1], self.kernel_size, stride=self.stride, padding=self.padding)
        self.conv3=nn.Conv2d(self.channel[1], self.channel[2], self.kernel_size, stride=self.stride, padding=self.padding)
        self.fc1=nn.Linear(view_size, self.n_hidden)
        self.mu = nn.Linear(n_hidden, latent_size)
        self.sigma = nn.Sequential(
            nn.Linear(n_hidden,latent_size),
            nn.Softplus())

#encoder forward function returns the parameters of the latent variables distribution: mean mu and deviation sigma
    def forward(self,inputs):
        output = self.activation(self.conv1(inputs))
        output = self.activation(self.conv2(output))
        output = self.activation(self.conv3(output))
        output = output.view(-1,output.shape[1]*output.shape[2]*output.shape[3])
        output = self.fc1(output)
        output = self.activation(output)
        mu = self.mu(output)
        sigma = self.sigma(output)
        return mu,sigma

class construct_decoder(nn.Module):
    def __init__(self, stride=2,latent_size=64, n_hidden=450,n_fc_layer=512,kernel_size=5,padding=2,channel=(32,32,16,1),activation=nn.Softplus()):
        super(construct_decoder, self).__init__()
        self.n_hidden = n_hidden
        self.activation = activation
        self.channel = channel
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.n_fc_layer=n_fc_layer
        self.fc1=nn.Linear(latent_size, self.n_hidden)
        self.fc2=nn.Linear(self.n_hidden, self.n_fc_layer)
        self.deconv1 = nn.ConvTranspose2d(self.channel[0], self.channel[1], self.kernel_size, stride=self.stride,
                                          padding=self.padding)
        self.deconv2 = nn.ConvTranspose2d(self.channel[1], self.channel[2], self.kernel_size, stride=self.stride,
                                          padding=self.padding, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(self.channel[2], self.channel[3], 5, stride=self.stride, padding=self.padding,
                                          output_padding=1)
#decoder forward: return the reconstructed x
    def forward(self,inputs):
        output = self.activation(self.fc1(inputs))
        output = self.activation(self.fc2(output))
        reshape = output.view((-1, self.channel[0], 4, 4))
        output = self.activation(self.deconv1(reshape))
        output = self.activation(
            self.deconv2(output))
        output = torch.sigmoid(
            self.deconv3(output))
        return output

def construct_encoder_decoder():
    # Encoder network
    encoder = construct_encoder()
    # Decoder network
    decoder = construct_decoder()
    return encoder, decoder


def binary_loss(x_tilde, x):
    return F.binary_cross_entropy(x_tilde, x, reduction='none').sum(dim = 0)

def multinomial_loss(x_logit, x,num_classes):
    batch_size = x.shape[0]
    # Reshape input
    x_logit = x_logit.view(batch_size, num_classes, x.shape[1], x.shape[2], x.shape[3])
    # Take softmax
    x_logit = F.log_softmax(x_logit, 1)
    # make integer class labels
    target = (x * (num_classes - 1)).long()
    # computes cross entropy over all dimensions separately:
    ce = F.nll_loss(x_logit, target, weight=None, reduction='none')
    return ce.sum(dim = 0)*100

def reconstruction_loss(x_tilde, x, num_classes=1, average=True):
    #here we use binary loss for MNIST dataset
    if (num_classes == 1):
        loss = binary_loss(x_tilde.flatten(), x.flatten())
    else:
        loss = multinomial_loss(x_tilde, x)
    if (average):
        loss = loss.sum() / x.size(0)
    return loss


# Main class for normalizing flow
class VAENormalizingFlow(VAE):

    def __init__(self, encoder, decoder, flow, encoder_dims=450, latent_dims=64):
        super(VAENormalizingFlow, self).__init__(encoder, decoder, encoder_dims, latent_dims)

        self.flow = flow
        if self.flow.flow_name=='normalizing_flow':
            self.flow_enc = nn.Linear(encoder_dims, self.flow.n_parameters())
            self.flow_enc.weight.data.uniform_(-0.01, 0.01)
        self.apply(self.init_parameters)


    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        if self.flow.flow_name=='normalizing_flow':
            flow_params = self.flow_enc(x)
        return mu, sigma, flow_params

    def latent(self, x, z_params):
        n_batch = x.size(0)
        # Split the encoded values to get flow parameters
        mu, sigma, flow_params = z_params
        # Re-parametrize a Normal distribution
        q = distrib.Normal(torch.zeros(mu.shape[1]), torch.ones(sigma.shape[1]))
        # Obtain our first set of latent points
        z_0 = (sigma * q.sample((n_batch,))) + mu
        # Update flows parameters
        self.flow.set_parameters(flow_params)
        # Complexify posterior with flows
        z_k, list_ladj = self.flow(z_0)
        # ln p(z_k)
        log_p_zk = (-0.5 * z_k * z_k)
        # ln q(z_0)
        log_q_z0 = -sigma.log() -0.5* (z_0 - mu) * (z_0 - mu)/(sigma**2)
        #  ln q(z_0) - ln p(z_k)
        logs = (log_q_z0 - log_p_zk).sum()
        # Add log determinants
        ladj = torch.cat(list_ladj)
        # ln q(z_0) - ln p(z_k) - sum[log det]
        logs -= torch.sum(ladj)
        return z_k, (logs / float(n_batch))



def train_vae(model, optimizer, train_loader, model_name='basic', epochs=10, plot_it=1, subsample=500, flatten=False,num_classes=1,batch_size=1,nin=784):
    # Losses curves
    losses = torch.zeros(epochs, 3)
    # Main optimization loop
    for it in range(epochs):
        # Update our beta
        #beta=1.0
        beta = 1. * (it / float(epochs))
        n_batch = 0.
        # Evaluate loss and backprop
        for batch_idx, (x, _) in enumerate(train_loader):
            if (batch_idx * batch_size) > subsample:
                break
            # Flatten input data
            if (flatten):
                x = x.view(-1, nin)
            # Pass through VAE
            x_tilde, loss_latent = model(x)
            # Compute reconstruction loss
            loss_recons = reconstruction_loss(x_tilde, x, num_classes)
            # Evaluate loss and backprop
            loss = loss_recons + beta*loss_latent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses[it, 0] += loss_recons.item()
            losses[it, 1] += loss_latent.item()
            losses[it, 2] += loss.item()
            n_batch += 1.
        losses[it, :] /= n_batch
        print(("Epoch:{:>4}, loss:{:>4.2f}").format(it+1,losses[it,2].item()))
    # visualize the training process
    plt.plot(range(epochs), (losses[:,2]).detach().numpy())
    plt.show()
    # save losses and model parameters
    np.save('normalizing_flow.npy',losses)
    torch.save(model.state_dict(),
               ("normalizing_flow.model"))
    return losses

def train_vae_particle_flow(encoder,decoder, optimizer, train_loader,n_lambda=3, model_name='basic', epochs=10, plot_it=1, subsample=500, flatten=False,num_classes=1,batch_size=1,nin=784):
    losses = torch.zeros(epochs, 3)
    for it in range(epochs):
        n_batch = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            if (batch_idx * batch_size) > subsample:
                break
            mu, sigma = encoder(x)
            q = distrib.Normal(torch.zeros(mu.shape[1]), torch.ones(sigma.shape[1]))
            # samples from Gaussian distribution N(mu,sigma)
            z_0 = ((sigma) * q.sample((1,))) + mu
            flow = particle_flow(decoder, mu, sigma, x, n_lambda=n_lambda)
            # z_0 flow to z_k
            z_k = flow(z_0)
            x_tilde = decoder(z_k)
            log_p_zk = (-0.5 * z_k * z_k).sum()
            # log(z_0|x)+sum(det)
            log_q_z0 = (-sigma.log() - 0.5 * (z_0 - mu) * (z_0 - mu) / (sigma ** 2)).sum()
            log_q_z0 = log_q_z0 - flow.gamma
            # losses of latent variables z_0 and z_k
            loss_latent = log_q_z0 - log_p_zk
            # reconstruction error
            loss_recons = reconstruction_loss(x_tilde, x, num_classes=1)
            loss = loss_recons + (loss_latent)
            optimizer.zero_grad()
            loss.backward()
            # update paramters
            optimizer.step()
            print('Epoch:', it + 1, 'iter:', batch_idx, 'loss:', loss.item(), 'reconstruction_loss:', loss_recons,
                  'log_q_z0:', log_q_z0, 'log_p_zk:', log_p_zk)
            losses[it, 0] += loss_recons.item()
            losses[it, 1] += loss_latent.item()
            losses[it, 2] += loss_recons.item() + loss_latent.item()
            n_batch += 1
        losses[it, :] /= n_batch
        print(("Epoch:{:>4}, loss:{:>4.2f}").format(it + 1, losses[it, 0].item() + losses[it, 1].item()))
    # visualize the training process
    plt.plot(range(epochs), (losses[:, 2]).detach().numpy())
    plt.show()
    # save losses and model parameters
    np.save('particle_flow_' + str(n_lambda) + '_intervals.npy', losses)
    torch.save(encoder.state_dict(),
               ("encoder_particle_flow.model"))
    torch.save(decoder.state_dict(),
               ("decoder_particle_flow.model"))
    return losses