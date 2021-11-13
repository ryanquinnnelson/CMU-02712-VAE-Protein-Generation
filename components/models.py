import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchvision


class Encoder(nn.Module):
    """
    https://avandekleut.github.io/vae/
    """

    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)  # 28x28 pixel image
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(512, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(512, 784)
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, z):
        z = self.linear1(z)
        z = self.relu1(z)
        z = self.linear2(z)
        z = self.sigmoid1(z)
        return z.reshape((-1, 1, 28, 28))


class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        #         self.N.loc = self.N.loc.cuda()  # move sampling to GPU
        #         self.N.scale = self.N.scale.cuda() # same
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.relu1(x)

        mu = self.linear2(x)
        log_sigma = self.linear3(x)
        sigma = torch.exp(log_sigma)

        z = mu + sigma * self.N.sample(mu.shape)

        # update KL divergence of N(mu,sigma) from N(0,1)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x
