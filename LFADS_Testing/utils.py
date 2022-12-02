import h5py
from jax import random
import numpy as onp # original numpy
import os

MAX_SEED_INT = 10000000


def split_data(data_b, train_fraction):
  train_data_offset = 0
  ndata = data_b.shape[0]
  eval_data_offset = int(train_fraction * ndata)
  train_data = data_b[train_data_offset:eval_data_offset]
  eval_data = data_b[eval_data_offset:]

  return train_data, eval_data


def spikify_data(data_bxtxn, rng, dt=1.0, max_firing_rate=100):
  spikes_e = []
  B, T, N = data_bxtxn.shape
  for data_txn in data_bxtxn:
    spikes = onp.zeros([T,N]).astype(onp.int)
    for n in range(N):
      f = data_txn[:,n]
      s = rng.poisson(f*max_firing_rate*dt, size=T)
      spikes[:,n] = s
    spikes_e.append(spikes)
  return onp.array(spikes_e)


def merge_losses_dicts(list_of_dicts):
  merged_d = {}
  d = list_of_dicts[0]
  for k in d:
    merged_d[k] = []
  for d in list_of_dicts:
    for k in d:
      merged_d[k].append(d[k])
  for k in merged_d:
    merged_d[k] = onp.array(merged_d[k])
  return merged_d


def average_lfads_batch(lfads_dict):
  avg_dict = {}
  for k in lfads_dict:
    avg_dict[k] = onp.mean(lfads_dict[k], axis=0)
  return avg_dict


def keygen(key, nkeys):
  keys = random.split(key, nkeys+1)
  return keys[0], (k for k in keys[1:])


def ensure_dir(file_path):
  """Make sure the directory exists, create if it does not."""
  directory = os.path.dirname(file_path)
  if not os.path.exists(directory):
    os.makedirs(directory)

    
def write_file(data_fname, data_dict, do_make_dir=False):
  try:
    ensure_dir(data_fname)
      
    with h5py.File(data_fname, 'w') as hf:
      for k in data_dict:
        hf.create_dataset(k, data=data_dict[k])
        # add attributes
  except IOError:
    print("Cannot write % for writing." % data_fname)
    raise


def read_file(data_fname):
  try:
    with h5py.File(data_fname, 'r') as hf:
      data_dict = {k: onp.array(v) for k, v in hf.items()}
      return data_dict
  except IOError:
    print("Cannot open %s for reading." % data_fname)
    raise
