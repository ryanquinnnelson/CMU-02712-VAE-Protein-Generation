import torch
import torch.nn as nn


class CustomCriterion1:
    """
    Loss derived from https://avandekleut.github.io/vae/
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

    def calculate_loss(self, x, x_hat, mu, sigma, epoch):
        # KL divergence between p(z|x) and N(0,1)
        # penalizes p(z|x) from being too far from standard normal
        kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 0.5).sum()

        # reconstruction loss
        square_err = ((x - x_hat) ** 2).sum()

        # combined loss
        loss = square_err + (kl * self.kl_weight)

        # update burn in
        self.update_kl_weight(epoch)

        return loss

    def __str__(self):
        e1 = f'CustomCriterion1: \n\tloss=(kl + mse)'
        e2 = f'\n\tkl = (sigma ** 2 + mu ** 2 - log(sigma) - 0.5)'
        e3 = f'\n\tmse = ((x - x_hat ** 2)'
        e4 = f'\n\tuse_burn_in:{self.use_burn_in}'
        e5 = f'\n\tdelta_burn_in:{self.delta_burn_in}'
        e6 = f'\n\tburn_in_start:{self.burn_in_start}'

        return e1 + e2 + e3 + e4 + e5 + e6


# class CustomCriterion2:
#
#     def __init__(self):
#         # self.burn_in = 0
#         #
#         # if its > 300 and burn_in_counter < 1.0:
#         #     burn_in_counter += 0.003
#         pass
#
#     def calculate_loss(self, x, x_hat, mu, sigma, epoch):
#         recon_loss = nn.binary_cross_entropy(x_hat, x,
#                                              size_average=False)  # by setting to false it sums instead of avg.
#         kl_loss = 0.5 * torch.sum(torch.exp(z_var) + mu ** 2 - 1. - z_var)
#         # kl_loss = kl_loss*burn_in_counter
#         loss = recon_loss + kl_loss
#
#         return loss
