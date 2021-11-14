import torch
import torch.nn as nn

class CustomCriterion1:

    def __init__(self):
        pass

    def calculate_loss(self, x, x_hat, mu, sigma):
        kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()

        # TODO: stop reshaping once no longer doing MNIST
        mse = ((x - x_hat.reshape((-1, 1, 28, 28))) ** 2).sum()

        loss = kl + mse

        return loss

    def __str__(self):
        e1 = f'CustomCriterion1: \nloss=(kl + mse)'
        e2 = f'\nkl = (sigma ** 2 + mu ** 2 - log(sigma) - 1/2)'
        e3 = f'\nmse = ((x - x_hat ** 2)'

        return e1 + e2 + e3


# class CustomCriterion2:
#
#     def __init__(self):
#         pass
#
#     def calculate_loss(self, x, x_hat, mu_sigma):
#         recon_loss = nn.binary_cross_entropy(x_hat, x,
#                                              size_average=False)  # by setting to false it sums instead of avg.
#         kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var)
#         kl_loss = kl_loss*burn_in_counter
#         loss = recon_loss + kl_loss
#
#         return loss
