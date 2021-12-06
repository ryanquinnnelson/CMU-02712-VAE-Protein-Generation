"""
Custom loss functions that combine multiple types of loss.
"""
__author__ = 'ryanquinnnelson'

import logging

import torch
import torch.nn as nn


class CustomCriterion1:
    """
    Uses Squared Error + KL divergence
    Loss derived from https://avandekleut.github.io/vae/. See README.md for more information.
    """

    def __init__(self, use_burn_in, delta_burn_in, burn_in_start):
        self.use_burn_in = use_burn_in
        self.delta_burn_in = delta_burn_in
        self.burn_in_start = burn_in_start

        # if KL loss should not use burn_in, set burn_in to 1.0 so value is never updated
        if use_burn_in:
            self.kl_weight = 0.0
        else:
            self.kl_weight = 1.0

    def update_kl_weight(self, epoch):

        # increment kl_weight if
        #   - epoch is greater than starting point for burn in
        #   - burn in is not already at 1.0
        if epoch >= self.burn_in_start and self.kl_weight <= (1.0 - self.delta_burn_in):
            self.kl_weight += self.delta_burn_in
            logging.info(f'kl_weight:{self.kl_weight}')

    def calculate_loss(self, x, x_hat, mu, sigma, i):
        # KL divergence between p(z|x) and N(0,1)
        # penalizes p(z|x) from being too far from standard normal
        kl_loss = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 0.5).sum()

        recon_loss = ((x - x_hat) ** 2).sum()

        if i == 0:
            logging.info(f'kl_loss:{kl_loss}')
            logging.info(f'recon_loss:{recon_loss}')

        # combined loss
        loss = recon_loss + (kl_loss * self.kl_weight)
        return loss

    def __str__(self):
        e1 = f'CustomCriterion1: \n\tloss=(kl + se)'
        e2 = f'\n\tkl = (sigma ** 2 + mu ** 2 - log(sigma) - 0.5)'
        e3 = f'\n\tmse = ((x - x_hat ** 2)'
        e4 = f'\n\tuse_burn_in:{self.use_burn_in}'
        e5 = f'\n\tdelta_burn_in:{self.delta_burn_in}'
        e6 = f'\n\tburn_in_start:{self.burn_in_start}'
        e7 = f'\n\tkl_weight:{self.kl_weight}'

        return e1 + e2 + e3 + e4 + e5 + e6 + e7


class CustomCriterion2:
    """
    Uses binary cross entropy + KL divergence
    Loss derived from https://github.com/psipred/protein-vae/. See README.md for more information.
    """

    def __init__(self, use_burn_in, delta_burn_in, burn_in_start):
        self.use_burn_in = use_burn_in
        self.delta_burn_in = delta_burn_in
        self.burn_in_start = burn_in_start

        # if KL loss should not use burn_in, set burn_in to 1.0 so value is never updated
        if use_burn_in:
            self.kl_weight = 0.0
        else:
            self.kl_weight = 1.0

    def update_kl_weight(self, epoch):

        # increment kl_weight if
        #   - epoch is greater than starting point for burn in
        #   - burn in is not already at 1.0
        if epoch >= self.burn_in_start and self.kl_weight <= (1.0 - self.delta_burn_in):
            self.kl_weight += self.delta_burn_in
            logging.info(f'kl_weight:{self.kl_weight}')

    def calculate_loss(self, x, x_hat, mu, sigma, i):

        # KL divergence between p(z|x) and N(0,1)
        # penalizes p(z|x) from being too far from standard normal
        kl_loss = (0.5 * sigma ** 2 + 0.5 * mu ** 2 - torch.log(sigma) - 0.5).sum()

        recon_loss = nn.functional.binary_cross_entropy(x_hat, x, size_average=False)  # sums instead of averaging

        if i == 0:
            logging.info(f'kl_loss:{kl_loss}')
            logging.info(f'recon_loss:{recon_loss}')

        # combined loss
        loss = recon_loss + (kl_loss * self.kl_weight)

        return loss

    def __str__(self):
        e1 = f'CustomCriterion2: \n\tloss=(kl + bce)'
        e2 = f'\n\tkl = (0.5 * sigma ** 2 + 0.5 * mu ** 2 - torch.log(sigma) - 0.5)'
        e3 = f'\n\tmse = binary_cross_entropy(x_hat, x)'
        e4 = f'\n\tuse_burn_in:{self.use_burn_in}'
        e5 = f'\n\tdelta_burn_in:{self.delta_burn_in}'
        e6 = f'\n\tburn_in_start:{self.burn_in_start}'
        e7 = f'\n\tkl_weight:{self.kl_weight}'

        return e1 + e2 + e3 + e4 + e5 + e6 + e7
