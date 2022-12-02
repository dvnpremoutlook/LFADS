

from __future__ import print_function, division, absolute_import

import matplotlib.pyplot as plt
import numpy as onp

from scipy import stats
from sklearn.decomposition import PCA

def plot_data_pca(data_dict):
  """Plot the PCA skree plot of the hidden units in the integrator RNN."""
  f = plt.figure()
  ndata, ntime, nhidden = data_dict['hiddens'].shape

  print('Number of data examples: ', ndata)
  print('Number of timesteps: ', ntime)
  print('Number of data dimensions: ', nhidden)
  pca = PCA(n_components=100)
  pca.fit(onp.reshape(data_dict['hiddens'], [ndata * ntime, nhidden]))

  plt.plot(onp.arange(1, 16), onp.cumsum(pca.explained_variance_ratio_)[0:15],
           '-o');
  plt.plot([1, 15], [0.95, 0.95])
  plt.xlabel('PC #')
  plt.ylabel('Cumulative Variance')
  plt.xlim([1, 15])
  plt.ylim([0.3, 1]);
  return f


def plot_data_example(input_bxtxu, hidden_bxtxn=None,
                      output_bxtxo=None, target_bxtxo=None, bidx=None):
  """Plot a single example of the data from the data integrator RNN."""
  if bidx is None:
    bidx = onp.random.randint(0, input_bxtxu.shape[0])
  ntoplot = 10
  ntimesteps = input_bxtxu.shape[1]
  f = plt.figure(figsize=(10,8))
  plt.subplot(311)
  plt.plot(input_bxtxu[bidx,:,0])
  plt.xlim([0, ntimesteps-1])
  plt.ylabel('Input')
  plt.title('Example %d'%bidx)
  if hidden_bxtxn is not None:
    plt.subplot(312)
    plt.plot(hidden_bxtxn[bidx, :, 0:ntoplot] + 0.25*onp.arange(0, ntoplot, 1), 'b')
    plt.ylabel('Hiddens')
    plt.xlim([0, ntimesteps-1]);
  plt.subplot(414)
  if output_bxtxo is not None:
    plt.plot(output_bxtxo[bidx,:,0].T, 'r');
    plt.xlim([0, ntimesteps-1]);
    plt.ylabel('Output / Targets')
    plt.xlabel('Time')
  if target_bxtxo is not None:
    plt.plot(target_bxtxo[bidx,:,0], 'k');
    plt.xlim([0, ntimesteps-1]);
  return f


def plot_data_stats(data_dict, data_bxtxn, data_dt):
  """Plot the statistics of the data integrator RNN data after spikifying."""
  print(onp.mean(onp.sum(data_bxtxn, axis=1)), "spikes/second")
  f = plt.figure(figsize=(12,4))
  plt.subplot(141)
  plt.hist(onp.mean(data_bxtxn, axis=1).ravel()/data_dt);
  plt.xlabel('spikes / sec')
  plt.subplot(142)
  plt.imshow(data_dict['hiddens'][0,:,:].T)
  plt.xlabel('time')
  plt.ylabel('neuron #')
  plt.title('Sample trial rates')
  plt.subplot(143);
  plt.imshow(data_bxtxn[0,:,:].T)
  plt.xlabel('time')
  plt.ylabel('neuron #')
  plt.title('spikes')
  plt.subplot(144)
  plt.stem(onp.mean(onp.sum(data_bxtxn, axis=1), axis=0));
  plt.xlabel('neuron #')
  plt.ylabel('spikes / sec');
  return f


def plot_losses(tlosses, elosses, sampled_every):
  """Plot the losses associated with training LFADS."""
  f = plt.figure(figsize=(15, 12))
  for lidx, k in enumerate(tlosses):
    plt.subplot(3, 2, lidx+1)
    tl = tlosses[k].shape[0]
    x = onp.arange(0, tl) * sampled_every
    plt.plot(x, tlosses[k], 'k')
    plt.plot(x, elosses[k], 'r')
    plt.axis('tight')
    plt.title(k)

  return f


def plot_priors(params):
  """Plot the parameters of the LFADS priors."""
  prior_dicts = {'ic' : params['ic_prior'], 'ii' : params['ii_prior']}
  pidxs = (pidx for pidx in onp.arange(1,12))
  f = plt.figure(figsize=(12,8))
  for k in prior_dicts:
    for j in prior_dicts[k]:
      plt.subplot(2,3,next(pidxs));
      data = prior_dicts[k][j]
      if "log" in j:
        data = onp.exp(data)
        j_title = j.strip('log')
      else:
        j_title = j
      plt.stem(data)
      plt.title(k + ' ' + j_title)
  return f


def plot_lfads(x_txd, avg_lfads_dict):
  """Plot the full state ofLFADS operating on a single example."""
  ld = avg_lfads_dict

  def remove_outliers(A, nstds=3):
    clip = nstds * onp.std(A)
    A_mean = onp.mean(A)
    A_show = onp.where(A < A_mean - clip, A_mean - clip, A)
    return onp.where(A_show > A_mean + clip, A_mean + clip, A_show)
    
  f = plt.figure(figsize=(12,12))

  rates = remove_outliers(onp.exp(ld['lograte_t']))
  plt.imshow(rates.T)
  plt.title('rates')    
  plt.savefig('TTK.png')

  return f

