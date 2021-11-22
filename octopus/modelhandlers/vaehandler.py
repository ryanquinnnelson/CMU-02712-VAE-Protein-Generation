"""
Handler for VAE models.
"""
__author__ = 'ryanquinnnelson'

import logging

from octopus.models import vae


class VaeHandler:

    def __init__(self, model_type, input_size, hidden_sizes, latent_dim, batch_normalization,
                 dropout):
        logging.info('Initializing VAE handler...')

        self.model_type = model_type
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.latent_dim = latent_dim
        self.batch_normalization = batch_normalization
        self.dropout = dropout

    def get_model(self):
        """
        Initialize the LSTM model based on model_type.

        :return (nn.Module): LSTM model
        """
        model = None

        if self.model_type == 'PaperVAE':
            model = vae.PaperVAE(self.input_size, self.hidden_sizes, self.latent_dim, self.batch_normalization,
                                 self.dropout)
        logging.info(f'Model initialized:\n{model}')
        return model
