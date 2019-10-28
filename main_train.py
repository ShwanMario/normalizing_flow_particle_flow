
from torchvision import datasets
from torchvision import transforms
from Flow import *
from vae import *
import argparse
import matplotlib.pyplot as plt
# Define grids of points (for later plots)

import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description="VAE with Normalizing Flow")
parser.add_argument("--batch_size",type=int,
                    default=1,
                    help="""batch size for training""")
parser.add_argument("--flow_length",type=int,
                    default=4,
                    help="""length of normalizing flow""")
parser.add_argument("--n_epochs",type=int,
                    default=10000,
                    help="""number of training epochs""")
parser.add_argument("--n_subsamples",type=int,
                    default=500,
                    help="""number of samples used in training""")
parser.add_argument("--learning_rate",type=float,
                    default=0.001,
                    help="""learning rate""")
args = parser.parse_args()

batch_size =args.batch_size
learning_rate=args.learning_rate
n_epochs=args.n_epochs
flow_length=args.flow_length

# Number of hidden and latent
n_hidden = 450
n_latent = 64

#transform the MNIST dataset to pytorch tensor
tens_t = transforms.ToTensor()
train_dset = datasets.MNIST('./data', train=True, download=True, transform=tens_t)
train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True)
test_dset = datasets.MNIST('./data', train=False, transform=tens_t)
test_loader = torch.utils.data.DataLoader(test_dset, batch_size=batch_size, shuffle=True)



fixed_batch, fixed_targets = next(iter(test_loader))

block_planar = [PlanarFlow]
# Create normalizing flow
flow = NormalizingFlow(dim=n_latent, blocks=block_planar, flow_length=flow_length, density=distrib.MultivariateNormal(torch.zeros(n_latent), torch.eye(n_latent)))
# Construct encoder and decoder
encoder, decoder = construct_encoder_decoder()
# Create VAE with planar flows
model_flow = VAENormalizingFlow(encoder, decoder, flow, n_hidden, n_latent)
# Create optimizer algorithm
optimizer = optim.Adam(model_flow.parameters(), lr=args.learning_rate)
# Launch our optimization
losses_flow = train_vae(model_flow, optimizer,  train_loader, fixed_batch, model_name='normalizing_flow', epochs=n_epochs,subsample=500)


