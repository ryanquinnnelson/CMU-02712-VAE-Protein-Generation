"""
VAE Model

Based on https://github.com/psipred/protein-vae/


"""

from collections import OrderedDict

import torch
import torch.nn as nn


class LinearBlock(nn.Module):

    def __init__(self, input_dim, output_dim, batch_normalization, dropout_rate):
        super(LinearBlock, self).__init__()
        self.batch_normalization = batch_normalization
        self.dropout_rate = dropout_rate

        # layers
        self.linear = nn.Linear(input_dim, output_dim)

        if batch_normalization:
            self.bn = nn.BatchNorm1d(output_dim)

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

        self.relu = nn.ReLU(inplace=True)

        # initial layer weights
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = self.linear(x)

        if self.batch_normalization:
            x = self.bn(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        x = self.relu(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, batch_normalization, dropout):
        super(MLP, self).__init__()

        sequence = []

        sizes = [input_size] + hidden_sizes + [output_size]

        # number of blocks is equal to the number of hidden layers
        for i in range(len(hidden_sizes) + 1):
            layer_name = 'block' + str(i + 1)
            linear_tuple = (layer_name, LinearBlock(sizes[i], sizes[i + 1], batch_normalization, dropout))
            sequence.append(linear_tuple)

        self.blocks = nn.Sequential(OrderedDict(sequence))

    def forward(self, x):
        x = self.blocks(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_sizes, output_size, batch_normalization, dropout):
        super(Decoder, self).__init__()

        self.mlp = MLP(latent_dim, hidden_sizes[:-1], hidden_sizes[-1], batch_normalization, dropout)
        self.linear = nn.Linear(hidden_sizes[-1], output_size)
        self.sigmoid = nn.Sigmoid()

        # initial layer weights
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = self.mlp(x)
        x = self.linear(x)
        x = self.sigmoid(x)

        return x


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_sizes, latent_dim, batch_normalization, dropout):
        super(Encoder, self).__init__()

        self.mlp = MLP(input_size, hidden_sizes[:-1], hidden_sizes[-1], batch_normalization, dropout)
        self.linear1 = nn.Linear(hidden_sizes[-1], latent_dim)
        self.linear2 = nn.Linear(hidden_sizes[-1], latent_dim)

        self.softplus = nn.Softplus()

        # initial layer weights
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # may be unnecessary
        x = self.mlp(x)
        mu = self.linear1(x)
        log_var = self.softplus(self.linear2(x))
        sigma = torch.exp(log_var / 2.0)  # See README.md for why we consider this to be std dev

        # sample from normal distribution
        eps = torch.randn(mu.shape)  # ~N(0,1) in the correct shape to multiply by sigma and add to mu
        z = mu + sigma * eps

        # move to gpu if necessary
        if 'cuda' in str(next(self.parameters()).device):
            z = z.to(device=torch.device('cuda'))

        return z, mu, sigma


class PaperVAE(nn.Module):
    def __init__(self, input_size, hidden_sizes, latent_dim, batch_normalization, dropout):
        super(PaperVAE, self).__init__()

        self.input_size = input_size
        self.batch_normalization = batch_normalization
        self.dropout = dropout
        self.hidden_sizes = hidden_sizes
        self.latent_dim = latent_dim

        # Decoder layers are the reverse of the Encoder
        reverse_hidden_sizes = hidden_sizes.copy()
        reverse_hidden_sizes.reverse()

        self.Encoder = Encoder(input_size, self.hidden_sizes, self.latent_dim, batch_normalization, dropout)
        self.Decoder = Decoder(self.latent_dim, reverse_hidden_sizes, input_size, batch_normalization,
                               dropout)

    def forward(self, x):
        z, mu, sigma = self.Encoder(x)
        x = self.Decoder(z)
        return x, mu, sigma
