from __future__ import print_function, division, absolute_import

import jax.numpy as np
from jax import grad, jit, vmap
from jax import random

import utils as utils

def poisson_log_likelihood(x, log_rate):
  return x * log_rate - np.exp(log_rate)


def diag_gaussian_sample(key, mean, logvar, varmin=1e-16):
  logvar_wm = np.log(np.exp(logvar) + varmin)
  return mean + np.exp(0.5*logvar_wm) * random.normal(key, mean.shape)


def diag_gaussian_log_likelihood(z, mean=0.0, logvar=0.0, varmin=1e-16):
  logvar_wm = np.log(np.exp(logvar) + varmin)
  return (-0.5 * (logvar + np.log(2*np.pi) +
                  np.square((z-mean)/( np.exp(0.5*(logvar_wm))))))


def kl_gauss_gauss(z_mean, z_logvar, prior_params, varmin=1e-16):
  z_logvar_wm = np.log(np.exp(z_logvar) + varmin)
  prior_mean = prior_params['mean']
  prior_logvar_wm = np.log(np.exp(prior_params['logvar']) + varmin)
  return (0.5 * (prior_logvar_wm - z_logvar_wm
                 + np.exp(z_logvar_wm - prior_logvar_wm)
                 + np.square((z_mean - prior_mean) / np.exp(0.5 * prior_logvar_wm))
                 - 1.0))

batch_kl_gauss_gauss = vmap(kl_gauss_gauss, in_axes=(0, 0, None, None))


def kl_gauss_ar1(key, z_mean_t, z_logvar_t, ar1_params, varmin=1e-16):
  ll = diag_gaussian_log_likelihood
  sample = diag_gaussian_sample
  nkeys = z_mean_t.shape[0]
  key, skeys = utils.keygen(key, nkeys)
  ar1_mean = ar1_params['mean']
  ar1_lognoisevar = np.log(np.exp(ar1_params['lognvar'] + varmin))
  phi = np.exp(-np.exp(-ar1_params['logatau']))
  logprocessvar = ar1_lognoisevar - (np.log(1-phi) + np.log(1+phi))
  z0 = sample(next(skeys), z_mean_t[0], z_logvar_t[0], varmin)
  logq = ll(z0, z_mean_t[0], z_logvar_t[0], varmin)
  logp = ll(z0, ar1_mean, logprocessvar, 0.0)
  z_last = z0
  for z_mean, z_logvar in zip(z_mean_t[1:], z_logvar_t[1:]):
    z = sample(next(skeys), z_mean, z_logvar, varmin)
    logq += ll(z, z_mean, z_logvar, varmin)
    logp += ll(z, ar1_mean + phi * z_last, ar1_lognoisevar, 0.0)
    z_last = z

  kl = logq - logp
  return kl

batch_kl_gauss_ar1 = vmap(kl_gauss_ar1, in_axes=(0, 0, 0, None, None))




def diagonal_gaussian_params(key, n, mean=0.0, var=1.0):
  return {'mean' : mean * np.ones((n,)),
          'logvar' : np.log(var) * np.ones((n,))}


def ar1_params(key, n, mean, autocorrelation_tau, noise_variance):
  return {'mean' : mean * np.ones((n,)),
          'logatau' : np.log(autocorrelation_tau * np.ones((n,))),
          'lognvar' : np.log(noise_variance) * np.ones((n,))}
