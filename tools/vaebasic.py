from __future__ import absolute_import, division
from __future__ import print_function
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/daniel/projects/sample4Acause/tools/')
from forwardmodels import *
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
from autograd.scipy.special import expit as sigmoid
from autograd.misc.flatten import flatten_func
import scipy as sp
from autograd import grad, jacobian
from autograd.misc.optimizers import adam
from scipy.stats import norm, multivariate_normal
from skimage import io

p = 9
d = 3

def postsample(params, E):
    """Samples c from the var. posterior given e parametrized by params"""
    draws = []
    for n in range(E.shape[0]):
        log_mu, log_tril_diag, tril_offdiag = unpack_gaussian_params( neural_net_predict(resdict['phi'],E[n,:].reshape(1,9)) )
        mu, tril_diag = np.exp(log_mu), np.exp(log_tril_diag)
        gen_tril = fill_tril(tril_diag.ravel(), tril_offdiag.ravel())
        draws.append((mu + np.dot(npr.normal(size=(1,2)), gen_tril.T)).ravel())
    return( np.exp( draws ))  

def compare(params,LAI,CHL,LMA,S):
    E = prosail_3d_L8(np.array([[LAI,CHL,LMA]])).reshape(1,9)
    return( np.array( [postsample(params,E).ravel() for s in range(S)] ) ) 

def prosail_3d_L8_exp(c):
    c = np.minimum(c,np.array([4, 5, 4.8])) 
    return prosail_3d_L8(np.exp(c))

def TrilGaussKL(mu0,S0tril,mu1,S1tril):
    n = np.shape(mu0)[0]
    S0 = np.dot(S0tril,S0tril.T)
    S1 = np.dot(S1tril,S1tril.T)
    S1inv = np.linalg.inv(S1)
    term1 = np.sum(np.diag(np.dot(S1inv,S0)))
    term2 = np.dot( ( mu1-mu0 ).T, np.dot( S1inv, ( mu1-mu0 ) ) ) 
    term3 = np.sum(  np.log(np.diag(S1tril)) ) - np.sum( np.log(np.diag(S0tril))  )
    return 0.5*( term1 + term2 + 2*term3 - np.float64(n) ) 

def lowerinds(dim):
    tot = 0
    inds=[]
    for i in list(reversed(range(1,dim))):
        tot = tot + i
        inds.append(tot)
    return [0]+inds

def fill_tril(diag, lower):
    # diagonal part
    diagmat = np.diag(diag)
    # lower part
    d = np.shape(diag)[0]
    li = lowerinds(d)
    lowmat = np.zeros([d,d])
    for i in range(d-1):
        lowmat = lowmat + np.diag( lower[li[i]:li[i+1]], -i-1)
    return diagmat + lowmat

def findiff(f, x0, in_d, out_d):
    """Finite differces especially for prosail"""
    delta = 0.000001
    x0 = x0.ravel()
    diffs = np.builtins.list([])
    for i in range(in_d):
        step = np.zeros(in_d)
        step[i] = delta
        xa = x0 - step
        xb = x0 + step
        diffs.append(  (f(xb.reshape(1, in_d)) - f(xa.reshape(1, in_d))).ravel() /(2*delta) )
    return np.array(diffs).T

def plotpost(resdict,E):
    N = E.shape[0]
    C = []
    for n in range(N):
        m,_ = unpack_gaussian_params( neural_net_predict(resdict['phi'], E[n,:].reshape(1,9) ) )
        C.append(m)
    return np.array(C).reshape(N,2)

def diag_gaussian_log_density(x, mu, log_std):
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)

def unpack_gaussian_params(params):
    # Params of a diagonal Gaussian.
    log_mu, log_tril_diag, tril_offdiag = params[:, :d], params[:, d:2*d], params[:, 2*d:]
    return log_mu, log_tril_diag, tril_offdiag

def sample_diag_gaussian(mean, log_std, rs):
    return rs.randn(*mean.shape) * np.exp(log_std) + mean

def bernoulli_log_density(targets, unnormalized_logprobs):
    # unnormalized_logprobs are in R
    # Targets must be -1 or 1
    label_probabilities = -np.logaddexp(0, -unnormalized_logprobs*targets)
    return np.sum(label_probabilities, axis=-1)   # Sum across pixels.

def relu(x):    return np.maximum(0, x)

def init_net_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a (weights, biases) tuples for all layers."""
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n)+1)      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def batch_normalize(activations):
    if activations.shape[0]==1:
        return activations
    else:
        mbmean = np.mean(activations, axis=0, keepdims=True)
        return (activations - mbmean) / (np.std(activations, axis=0, keepdims=True) + 1)

def neural_net_predict(params, inputs):
    """Params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       Applies batch normalization to every layer but the last."""
    for W, b in params[:-1]:
        outputs = batch_normalize(np.dot(inputs, W) + b)  # linear transformation
        inputs = relu(outputs)                       # nonlinear transformation
    outW, outb = params[-1]
    outputs = np.dot(inputs, outW) + outb
    return outputs

