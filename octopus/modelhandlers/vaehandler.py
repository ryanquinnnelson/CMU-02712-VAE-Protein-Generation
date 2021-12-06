"""
Handler for VAE models.
"""
__author__ = 'ryanquinnnelson'

import logging

from octopus.models import vae


class VaeHandler:

    def __init__(self, model_type, input_size, hidden_sizes, latent_dim, batch_normalization,
                 dropout):
        """
        Initialize VaeHandler.

        :param model_type (str): Type of model to initialize
        :param input_size (int): Dimension of features used as input
        :param hidden_sizes (List): List of the dimension for each of the intermediate hidden layers. If list contains more than one value, multiple layers will be added to the model, in the order given in the list. At least one hidden layer is required.
        :param latent_dim (int): Dimension of the latent feature space.
        :param batch_normalization (Boolean): If True, performs batch normalization after each intermediate hidden layer.
        :param dropout (float): Percent of dropout in each of the intermediate hidden layers in the model. Use 0.0 to avoid dropout.

        """
        logging.info('Initializing VAE handler...')

        self.model_type = model_type
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.latent_dim = latent_dim
        self.batch_normalization = batch_normalization
        self.dropout = dropout

    def get_model(self):
        """
        Initialize the VAE model based on model_type.

        :return (nn.Module): VAE model
        """
        model = None

        if self.model_type == 'PaperVAE':
            model = vae.PaperVAE(self.input_size, self.hidden_sizes, self.latent_dim, self.batch_normalization,
                                 self.dropout)
        logging.info(f'Model initialized:\n{model}')
        return model
