# Inference over Radiative Transfer Models using Variational and Expectation Maximization Methods

Code to reproduce PROSAIL experiment and source code for performing approximate bayesian inference over a forward model using:
- Monte Carlo Expectation Maximization
- Variational inference assuming Gaussian variational posterior parametrized by a neural network. This is similar to the work by [Kingma et al.](https://arxiv.org/abs/1312.6114), except we use a physics model in the likelihood in stead of another neural network.
