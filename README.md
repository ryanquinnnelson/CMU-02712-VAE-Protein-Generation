# CMU-02712-PROJECT

Fall 2021 Biological Modeling and Simulation - Group Project

## On the differences between VAE versions

I compared two versions of Variational Autoencoders:

1) https://avandekleut.github.io/vae/;
2) https://github.com/psipred/protein-vae/.

### Standard Deviation vs Variance
I found there were differences in how the authors interpreted the output of the second linear layer in the Encoder. 

1) Assumes this value is the log of the std dev: log(sigma).
2) Assumes this value is the log of the variance: log(sigma^2)

This can be determined by how each uses this value to obtain z (called sample in 2).

1) z = mu + exp(log(sigma)) * N(0,1)
2) z = mu + exp(log(sigma^2) / 2) * N(0,1)

### Calculating KL Loss
I found there were differences in how the authors calculated KL divergence.

1) KL = (1.0 * sigma^2 + 1.0 * mu^2 - log(sigma) - 0.5).sum()
2) KL = (0.5 * sigma^2 + 0.5 * mu^2 - log(sigma) - 0.5).sum()

Version 2 adds half as much of each of the first two terms.


