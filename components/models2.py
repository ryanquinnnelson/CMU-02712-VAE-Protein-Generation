"""
VAE Model

Based on https://github.com/psipred/protein-vae/blob/master/fold_gen/grammar_VAE_pytorch.py


"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class LinearBlock(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearBlock, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()

        self.block1 = LinearBlock(input_size, hidden_sizes[0])
        self.block2 = LinearBlock(hidden_sizes[0], hidden_sizes[1])
        self.block3 = LinearBlock(hidden_sizes[1], output_size)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


# TODO: understand why authors added 1273 to input_size and subtracted 1273 from output_size for Decoder
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Decoder, self).__init__()

        self.mlp = MLP(input_size, hidden_sizes[:-1], hidden_sizes[-1])
        self.linear = nn.Linear(hidden_sizes[-1], output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.mlp(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x.reshape((-1, 1, 28, 28)) # temp for example


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Encoder, self).__init__()

        self.mlp = MLP(input_size, hidden_sizes[:-1], hidden_sizes[-1])
        self.linear1 = nn.Linear(hidden_sizes[-1], output_size)
        self.linear2 = nn.Linear(hidden_sizes[-1], output_size)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.mlp(x)
        mu = self.linear1(x)
        log_var = self.softplus(self.linear2(x))
        sigma = torch.exp(log_var / 2.0)  # alt model doesn't divide by 2

        # sample from normal distribution
        # mu.shape should equal (batch_size, output_size)
        # print(mu.shape)
        eps = torch.randn(mu.shape)  # ~N(0,1)
        z = mu + sigma * eps

        # move to gpu if necessary
        # print(f'next(self.parameters()).device:{next(self.parameters()).device}')
        if 'cuda' in str(next(self.parameters()).device):
            z = z.to(device=torch.device('cuda'))

        return z, mu, sigma


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(VariationalAutoencoder, self).__init__()

        # Decoder layers are the reverse of the Encoder
        reverse_hidden_sizes = hidden_sizes.copy()
        reverse_hidden_sizes.reverse()

        self.Encoder = Encoder(input_size, hidden_sizes[:-1], hidden_sizes[-1])
        self.Decoder = Decoder(reverse_hidden_sizes[0], reverse_hidden_sizes[1:], input_size)

    def forward(self, x):
        z, mu, sigma = self.Encoder(x)
        x = self.Decoder(z)
        return x, mu, sigma
