# CMU-02712-PROJECT

Fall 2021 Biological Modeling and Simulation - Group Project

## On the differences between VAE versions

I compared two versions of Variational Autoencoders: (1) https://avandekleut.github.io/vae/; (
2) https://github.com/psipred/protein-vae/. I found there were differences in how the authors interpreted the output of
the second linear layer in the Encoder. (1) determined this value to be `\sigma`
