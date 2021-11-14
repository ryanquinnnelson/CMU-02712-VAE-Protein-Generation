import torch
import torch.nn as nn


class CustomCriterion1:
    """
    Loss derived from https://avandekleut.github.io/vae/
    """

    def __init__(self):
        pass

    def calculate_loss(self, x, x_hat, mu, sigma):

        # KL divergence between p(z|x) and N(0,1)
        # penalizes p(z|x) from being too far from standard normal
        kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 0.5).sum()

        # reconstruction loss
        square_err = ((x - x_hat) ** 2).sum()

        loss = kl + square_err

        return loss

    def __str__(self):
        e1 = f'CustomCriterion1: \nloss=(kl + mse)'
        e2 = f'\nkl = (sigma ** 2 + mu ** 2 - log(sigma) - 0.5)'
        e3 = f'\nmse = ((x - x_hat ** 2)'

        return e1 + e2 + e3

class CustomCriterion2:

    def __init__(self):
        # self.burn_in = 0
        #
        # if its > 300 and burn_in_counter < 1.0:
        #     burn_in_counter += 0.003
        pass

    def calculate_loss(self, x, x_hat, mu, sigma):
        recon_loss = nn.binary_cross_entropy(x_hat, x,
                                             size_average=False)  # by setting to false it sums instead of avg.
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + mu ** 2 - 1. - z_var)
        # kl_loss = kl_loss*burn_in_counter
        loss = recon_loss + kl_loss

        return loss
