"""
Everything related to Variational Autoencoder models.

Based on https://github.com/psipred/protein-vae/ with additional inspiration from https://avandekleut.github.io/vae/
"""
__author__ = 'ryanquinnnelson'

import logging
from collections import OrderedDict

import torch
import torch.nn as nn


class LinearBlock(nn.Module):
    """
    Define an intermediate linear hidden layer with optional batch normalization and dropout.
    Use ReLU as the activation function. Initialize layer weights using Xavier Uniform initialization.
    """

    def __init__(self, input_dim, output_dim, batch_normalization, dropout_rate):
        """
        Initialize LinearBlock.

        :param input_dim (int):  Dimension of features used as input
        :param output_dim (int): Dimension of features used as output
        :param batch_normalization (Boolean): If True, performs batch normalization after the linear layer
        :param dropout_rate (float): Percent of dropout in the layer. Use 0.0 to avoid dropout.
        """
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
        """
        Forward pass through linear layer.

        :param x (torch.Tensor): layer input
        :return (torch.Tensor): layer output
        """
        x = self.linear(x)

        if self.batch_normalization:
            x = self.bn(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        x = self.relu(x)
        return x


class MLP(nn.Module):
    """
    Define a multilayer perceptron that represents all intermediate hidden layers in either the Encoder or Decoder.

    """

    def __init__(self, input_size, hidden_sizes, output_size, batch_normalization, dropout):
        """
        Initialize MLP.

        :param input_size (int): Dimension of features used as input
        :param hidden_sizes (List): List of the dimension for each of the intermediate hidden layers. If list contains more than one value, multiple layers will be added to the model, in the order given in the list. At least one hidden layer is required.
        :param output_size (int): Dimension of the latent feature space.
        :param batch_normalization (Boolean): If True, performs batch normalization after each intermediate hidden layer.
        :param dropout (float): Percent of dropout in each of the intermediate hidden layers in the model. Use 0.0 to avoid dropout.
        """
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
        """
        Forward pass through the MLP.

        :param x (torch.Tensor): MLP input. (input_dim x 1)
        :return (torch.Tensor): MLP output. (output_dim x 1)
        """
        x = self.blocks(x)
        return x


class Decoder(nn.Module):
    """
    Define a VAE Decoder with one or more hidden linear layers, followed by a linear layer representing VAE output,
    followed by a sigmoid activation function.
    """

    def __init__(self, latent_dim, hidden_sizes, output_size, batch_normalization, dropout):
        """
        Initialize Decoder.

        :param latent_dim (int): Dimension of the latent feature space.
        :param hidden_sizes (List): List of the dimension for each of the intermediate hidden layers. If list contains more than one value, multiple layers will be added to the model, in the order given in the list. At least one hidden layer is required.
        :param output_size (int): Dimension of the feature vector used as input to the VAE.
        :param batch_normalization (Boolean): If True, performs batch normalization after each intermediate hidden layer.
        :param dropout (float): Percent of dropout in each of the intermediate hidden layers in the model. Use 0.0 to avoid dropout.
        """
        super(Decoder, self).__init__()

        self.mlp = MLP(latent_dim, hidden_sizes[:-1], hidden_sizes[-1], batch_normalization, dropout)
        self.linear = nn.Linear(hidden_sizes[-1], output_size)
        self.sigmoid = nn.Sigmoid()

        # initial layer weights
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        """
        Forward pass through decoder. Reconstructs feature vector from latent vector.

        :param x (torch.Tensor): Decoder input (latent_dim x 1)
        :return (torch.Tensor): Decoder output (feature_dim x 1)
        """
        x = self.mlp(x)
        x = self.linear(x)
        x = self.sigmoid(x)

        return x


class Encoder(nn.Module):
    """
    Define an Encoder with a linear input layer, followed by one or more intermediate hidden layers,
    followed by parallel mean and standard deviation linear hidden layers, followed by linear layer
    representing the latent features. Linear layer representing standard deviation is followed by a
    SoftPlus activation layer.
    """

    def __init__(self, input_size, hidden_sizes, latent_dim, batch_normalization, dropout):
        """
        Initialize Encoder.

        :param input_size (int): Dimension of the feature space.
        :param hidden_sizes (List): List of the dimension for each of the intermediate hidden layers. If list contains more than one value, multiple layers will be added to the model, in the order given in the list. At least one hidden layer is required.
        :param latent_dim (int): Dimension of the latent feature space.
        :param batch_normalization (Boolean): If True, performs batch normalization after each intermediate hidden layer.
        :param dropout (float): Percent of dropout in each of the intermediate hidden layers in the model. Use 0.0 to avoid dropout.
        """
        super(Encoder, self).__init__()

        self.mlp = MLP(input_size, hidden_sizes[:-1], hidden_sizes[-1], batch_normalization, dropout)
        self.linear1 = nn.Linear(hidden_sizes[-1], latent_dim)
        self.linear2 = nn.Linear(hidden_sizes[-1], latent_dim)

        self.softplus = nn.Softplus()

        # initial layer weights
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x):
        """
        Perform a forward pass through Encoder layers.

        :param x (torch.Tensor): Input to VAE model. (feature_dim x 1)
        :return (torch.Tensor, torch.Tensor, torch.Tensor): Tuple representing latent space, mean layer, and standard deviation layer. All three have a shape (latent_dim x 1).
        """
        x = self.mlp(x)
        mu = self.linear1(x)
        log_var = self.softplus(self.linear2(x))
        sigma = torch.exp(log_var / 2.0)  # See README.md for why we consider this to be std dev

        # sample from normal distribution
        eps = torch.randn(mu.shape)  # ~N(0,1) in the correct shape to multiply by sigma and add to mu

        # move to gpu if necessary
        if 'cuda' in str(next(self.parameters()).device):
            eps = eps.to(device=torch.device('cuda'))

        z = mu + sigma * eps

        # move to gpu if necessary
        if 'cuda' in str(next(self.parameters()).device):
            z = z.to(device=torch.device('cuda'))

        return z, mu, sigma


class PaperVAE(nn.Module):
    """
    Define a Variational Autoencoder model based on Greener et. al (https://www.nature.com/articles/s41598-018-34533-1).
    """

    def __init__(self, input_size, hidden_sizes, latent_dim, batch_normalization, dropout):
        """
        Initialize PaperVAE.

        :param input_size (int): Dimension of the feature space.
        :param hidden_sizes (List): List of the dimension for each of the intermediate hidden layers. If list contains more than one value, multiple layers will be added to the model, in the order given in the list. At least one hidden layer is required.
        :param latent_dim (int): Dimension of the latent feature space.
        :param batch_normalization (Boolean): If True, performs batch normalization after each intermediate hidden layer.
        :param dropout (float): Percent of dropout in each of the intermediate hidden layers in the model. Use 0.0 to avoid dropout.
        """
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

    def forward(self, i, x):
        """
        Forward pass through VAE.

        :param i (int): The number of the training epoch. Used to control when debugging output is produced.
        :param x (torch.Tensor): Input to model. (feature_dim x 1)
        :return (torch.Tensor, torch.Tensor, torch.Tensor): Output of model. Tuple representing reconstructed features, mean layer, and standard deviation layer. Mean and std dev layers have a shape (latent_dim x 1). Reconstructed output has shape (feature_dim x 1).
        """
        z, mu, sigma = self.Encoder(x)
        x = self.Decoder(z)
        return x, mu, sigma
