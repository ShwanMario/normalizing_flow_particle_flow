import torch
import torch.nn as nn

import torch.distributions as distrib
import torch.distributions.transforms as transform

# Imports for plotting
import numpy as np


class Flow(transform.Transform, nn.Module):

    def __init__(self):
        transform.Transform.__init__(self)
        nn.Module.__init__(self)

    # Init all parameters
    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)

    # Hacky hash bypass
    def __hash__(self):
        return nn.Module.__hash__(self)
#split the interval [0,1] into n intervals
def divide(q,n):
    i=[]
    n1=(1-q)/(1-q**n)
    i.append(n1)
    for k in range(n-1):
        n1*=q
        i.append(n1)
    return np.array(i)

class particle_flow(Flow):
    def __init__(self,decoder,mu,sigma,x,n_samples=1,q=6/5,n_lambda=3,flow_name='particle_flow'):
        super(particle_flow, self).__init__()
        self.n_samples=n_samples
        self.intervals=divide(q,n_lambda)
        self.decoder=decoder
        self.mu=mu
        self.sigma=sigma
        self.x=x
        self.flow_name=flow_name
#get the jacobian matrix of x_tilde w.r.t. z
    def get_jacobian(self,z,x_tilde):
        z_repeat=z.repeat(x_tilde.shape[-1],1,1)
        x_tilde_repeat=self.decoder(z_repeat)
        x_tilde_repeat=x_tilde_repeat.reshape((x_tilde.shape[-1],x_tilde.shape[-1]))
        mat_repeat=torch.eye(x_tilde.shape[0])
        z_repeat.retain_grad()
        x_tilde_repeat.backward(mat_repeat,retain_graph=True)
        return z_repeat.grad.data
#particle flow
    def _call(self, z):
        #covariance matrix of latent variables z
        p=torch.diag(self.sigma[0])
        #reconstructed x before particle flow
        x_tilde=self.decoder(z)
        x_flatten_tilde = x_tilde.flatten()
        bar_eta=self.mu.reshape((self.mu.shape[0],-1,self.mu.shape[1]))
        #covariance matrix of x_tilde, 784x784
        r = torch.diag(torch.max(x_flatten_tilde,1e-2*torch.ones(x_flatten_tilde.shape))* torch.max(1 - x_flatten_tilde,1e-2*torch.ones(x_flatten_tilde.shape)))#to avoid much too small elements of matrix r, here we set the minimum value as 1e-2
        #determinant of particle flow
        gamma=0
        alpha=1
        #particles before flow
        eta_1=z
        #intervals
        lambda_1=0
        #start to flow
        for j in range(self.intervals.shape[0]):
            lambda_1 = lambda_1 + self.intervals[j]
            h_eta=self.decoder(bar_eta)
            h_eta=h_eta.flatten()
            H=self.get_jacobian(bar_eta,h_eta).squeeze()
            bar_eta=bar_eta.squeeze(1)
            square_mat=lambda_1*H@p@(H.transpose(1,0))+r
            A_j_lambda=-1/2*p@H.transpose(1,0)@square_mat.inverse()@H
            temp=(torch.eye(bar_eta.shape[-1])+lambda_1*A_j_lambda)@p@H.transpose(1,0)@r.inverse()@self.x.reshape((784,1))+A_j_lambda@self.mu.transpose(1,0)
            b_j_lambda=(torch.eye(bar_eta.shape[-1])+2*lambda_1*A_j_lambda)@temp
            bar_eta=(bar_eta.transpose(1,0)+self.intervals[j]*(A_j_lambda@bar_eta.transpose(1,0)+b_j_lambda)).transpose(1,0)
            eta_1=(eta_1.transpose(1,0)+self.intervals[j]*(A_j_lambda@eta_1.transpose(1,0)+b_j_lambda)).transpose(1,0)
            alpha_mat = torch.eye(A_j_lambda.shape[0]) + self.intervals[j] * A_j_lambda
            alpha=alpha* torch.abs(torch.det(alpha_mat))
        #end of flow
        gamma =gamma+ torch.log(alpha)
        self.gamma=gamma
        #return the particles after flow
        return eta_1
#the determinant of particle flow
    def log_abs_det_jacobian(self,z):
        return self.gamma
class PlanarFlow(Flow):

    def __init__(self, dim,flow_name='normalizing_flow'):
        super(PlanarFlow, self).__init__()
        self.weight = []
        self.scale = []
        self.bias = []
        self.dim = dim
        self.flow_name=flow_name

    def _call(self, z):
        z = z.unsqueeze(2)
        f_z = torch.bmm(self.weight, z) + self.bias
        return (z + self.scale * torch.tanh(f_z)).squeeze(2)

    def log_abs_det_jacobian(self, z):
        z = z.unsqueeze(2)
        f_z = torch.bmm(self.weight, z) + self.bias
        psi = self.weight * (1 - torch.tanh(f_z) ** 2)
        det_grad = 1 + torch.bmm(psi, self.scale)
        return torch.log(det_grad.abs() + 1e-9)

    def set_parameters(self, p_list):
        self.weight = p_list[:, :self.dim].unsqueeze(1)
        self.scale = p_list[:, self.dim:self.dim * 2].unsqueeze(2)
        self.bias = p_list[:, self.dim * 2].unsqueeze(1).unsqueeze(2)

    def n_parameters(self):
        return 2 * self.dim + 1

class RadialFlow(Flow):

    def __init__(self, dim,flow_name='normalizing_flow'):
        super(RadialFlow, self).__init__()
        self.z0 = nn.Parameter(torch.Tensor(1, dim))
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(1))
        self.dim = dim
        self.init_parameters()
        self.flow_name=flow_name

    def _call(self, z):
        r = torch.norm(z - self.z0, dim=1).unsqueeze(1)
        h = 1 / (self.alpha + r)
        return z + (self.beta * h * (z - self.z0))

    def log_abs_det_jacobian(self, z):
        r = torch.norm(z - self.z0, dim=1).unsqueeze(1)
        h = 1 / (self.alpha + r)
        hp = - 1 / (self.alpha + r) ** 2
        bh = self.beta * h
        det_grad = ((1 + bh) ** self.dim - 1) * (1 + bh + self.beta * hp * r)
        return torch.log(det_grad.abs() + 1e-9)
    def set_parameters(self, p_list):
        self.weight = p_list[:, :self.dim].unsqueeze(1)
        self.scale = p_list[:, self.dim:self.dim+1].unsqueeze(2)
        self.bias = p_list[:, self.dim+1:self.dim+2].unsqueeze(1).unsqueeze(2)

    def n_parameters(self):
        return self.dim+2


# Main class for normalizing flow
class NormalizingFlow(nn.Module):

    def __init__(self, dim, blocks,  density,flow_name='normalizing_flow',flow_length=1,encoder=None,decoder=None):
        super().__init__()
        biject = []
        self.flow_name = flow_name
        self.n_params = []
        if flow_name=='normalizing_flow':
            for f in range(flow_length):
                for b_flow in blocks:
                    cur_block = b_flow(dim)
                    biject.append(cur_block)
                    self.n_params.append(cur_block.n_parameters())
        else:
            for b_flow in blocks:
                biject.append(b_flow())
        self.transforms = transform.ComposeTransform(biject)
        self.bijectors = nn.ModuleList(biject)
        self.base_density = density
        self.final_density = distrib.TransformedDistribution(density, self.transforms)
        self.log_det = []
        self.dim = dim
        self.encoder=encoder
        self.decoder=decoder


    def forward(self, z):
        self.log_det = []
        # Applies series of flows
        for b in range(len(self.bijectors)):
            self.log_det.append(self.bijectors[b].log_abs_det_jacobian(z))
            z = self.bijectors[b](z)
        return z, self.log_det

    def n_parameters(self):
        return sum(self.n_params)

    def set_parameters(self, params):
        param_list = params.split(self.n_params, dim=1)
        # Applies series of flows
        for b in range(len(self.bijectors)):
            self.bijectors[b].set_parameters(param_list[b])





