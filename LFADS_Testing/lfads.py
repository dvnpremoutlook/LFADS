
import datetime
import h5py
import jax.numpy as np
from jax import random
from jax.experimental import optimizers
from jax.config import config
import matplotlib.pyplot as plt
import numpy as onp  
import scipy.signal
import scipy.stats
import os
import sys
import time

import lfads_tutorial.lfads as lfads
import lfads_tutorial.plotting as plotting
import lfads_tutorial.utils as utils
from lfads_tutorial.optimize import optimize_lfads, get_kl_warmup_fun
from configparser import ConfigParser

onp_rng = onp.random.RandomState(seed=0) 

import numpy
import cv2

lfads_dir = './'  
rnn_type = 'lfads'
task_type = 'integrator'
data_dir = os.path.join(lfads_dir, 'data/')
output_dir = os.path.join(lfads_dir, 'output/')
figure_dir = os.path.join(lfads_dir, os.path.join(output_dir, 'figures/'))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)


data_dt = 1.0/25.0 
max_firing_rate = 80      
train_fraction = 0.95
renormed_fun = lambda x : (x + 1) / 2.0
frames = []
segmented_frames = []


# original_images = './Segmentation_dataset/1803290511/clip_00000000/'
original_images = './Small_Demo_Dataset/Images/'
# original_images = '/home/013057356/LFADS/LFADS_Testing/LFADS_dataset/LFADS_Full_Human_Dataset/proper_images/'
for filename in os.scandir(original_images):
    if filename.is_file():
      img = cv2.imread(filename.path)
      # Z = img.reshape((-1,3))
      # Z = onp.float32(Z)
      # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
      # K = 8
      # ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
      # center = onp.uint8(center)
      # res = center[label.flatten()]
      # res2 = res.reshape((img.shape))

      img = cv2.resize(img, (50, 50))
      frames.append(img)
print('original_images Done')
# segmentated_images = './Segmentation_dataset/1803290511/matting_00000000/'
segmentated_images = './Small_Demo_Dataset/2ndmatting/'
# segmentated_images = '/home/013057356/LFADS/LFADS_Testing/LFADS_dataset/LFADS_Full_Human_Dataset/matting_main/'
for filename in os.scandir(segmentated_images):
    if filename.is_file():
      img = cv2.imread(filename.path)
      # Z = img.reshape((-1,3))
      # Z = onp.float32(Z)
      # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
      # K = 8
      # ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
      # center = onp.uint8(center)
      # res = center[label.flatten()]
      # res2 = res.reshape((img.shape))

      img = cv2.resize(img, (50, 50))
      segmented_frames.append(img)
print('segmentated_images Done') 
data_bxtxn = numpy.stack(frames, axis=0) 
data_bxtxn = data_bxtxn[:, :, :, 0]
nexamples, ntimesteps, data_dim = data_bxtxn.shape
train_data, eval_data = utils.split_data(data_bxtxn,train_fraction=train_fraction)
eval_data_offset = int(train_fraction * data_bxtxn.shape[0])



segmented_data_bxtxn = numpy.stack(segmented_frames, axis=0) 
segmented_data_bxtxn = segmented_data_bxtxn[:, :, :, 0]
segmented_nexamples, segmented_ntimesteps, segmented_data_dim = segmented_data_bxtxn.shape
segmented_train_data, segmented_eval_data = utils.split_data(segmented_data_bxtxn,train_fraction=train_fraction)
segmented_eval_data_offset = int(train_fraction * segmented_data_bxtxn.shape[0])


my_example_bidx = eval_data_offset + 0
my_example_hidx = 0
scale = max_firing_rate * data_dt
my_signal_spikified = data_bxtxn[my_example_bidx, :, my_example_hidx]
plt.stem(my_signal_spikified);

nfilt = 3
my_filtered_spikes = scipy.signal.filtfilt(onp.ones(nfilt)/nfilt, 1, my_signal_spikified)
# plt.plot(my_signal, 'r');
plt.plot(my_filtered_spikes);
plt.title("This looks terrible");
plt.legend(('True rate', 'Filtered spikes'));

import sklearn
ncomponents = 50
full_pca = sklearn.decomposition.PCA(ncomponents)
full_pca.fit(onp.reshape(data_bxtxn, [-1, data_dim]))

plt.stem(full_pca.explained_variance_)
plt.title('Those top 2 PCs sure look promising!');

ncomponents = 2
pca = sklearn.decomposition.PCA(ncomponents)
pca.fit(onp.reshape(data_bxtxn[0:eval_data_offset,:,:], [-1, data_dim]))
my_example_pca = pca.transform(data_bxtxn[my_example_bidx,:,:])
my_example_ipca = pca.inverse_transform(my_example_pca)

plt.plot(my_example_ipca[:,my_example_hidx])
plt.legend(('True rate', 'PCA smoothed spikes'))
plt.title('This a bit better.');

data_dim = train_data.shape[2] 
ntimesteps = train_data.shape[1] 

config_object = ConfigParser()
config_object.read("config.ini")

#Get the password
params = config_object["hyperparameters"]
batch_size = 128
# batch_size =              int(params["batch_size"])     
# enc_dim =                 int(params["enc_dim"])         
# con_dim =                 int(params["con_dim"])        
# ii_dim =                  int(params["ii_dim"])            
# gen_dim =                 int(params["gen_dim"])         
# factors_dim =             float(params["factors_dim"])      
# var_min =                 float(params["var_min"]) 
# l2reg =                   float(params["l2reg"])
# ic_prior_var =            float(params["ic_prior_var"])
# ar_mean =                 float(params["ar_mean"])  
# ar_autocorrelation_tau =  float(params["ar_autocorrelation_tau"])
# ar_noise_variance =       float(params["ar_noise_variance"])
# num_batches =             int(params["num_batches"])
# print_every =             int(params["print_every"])
# step_size =               float(params["step_size"])
# decay_factor =            float(params["decay_factor"])
# decay_steps =             int(params["decay_steps"])
# keep_rate =               float(params["keep_rate"])
# max_grad_norm =           float(params["max_grad_norm"])
# kl_warmup_start =         float(params["kl_warmup_start"])
# kl_warmup_end =           float(params["kl_warmup_end"])
# kl_min =                  float(params["kl_min"])
# kl_max =                  float(params["kl_max"])
# print(enc_dim)      
# print(con_dim  )    
# print(ii_dim      )      
# print(gen_dim      ) 
# print(factors_dim   )

# print(var_min)
# print(l2reg)
# print(ic_prior_var)
# print(ar_mean)
# print(ar_autocorrelation_tau)
# print(ar_noise_variance)
# print(num_batches)
# print(print_every)
# print(step_size)
# print(decay_factor)
# print(decay_steps)
# print(keep_rate)
# print(max_grad_norm)
# print(kl_warmup_start)
# print(kl_warmup_end)
# print(kl_min)
# print(kl_max)
# enc_dim = 256         
# con_dim = 256        
# ii_dim = 1            
# gen_dim = 256         
# factors_dim = 64      
# var_min = 0.01 
# l2reg = 0.0002
# ic_prior_var = 0.01 
# ar_mean = 0.0       
# ar_autocorrelation_tau = 1.0 
# ar_noise_variance = 0.1 
# num_batches = 5000         
# print_every = 50

# step_size = 0.06
# decay_factor = 0.99999 
# decay_steps = 1 
# keep_rate = 0.97 
# max_grad_norm = 10.0
# kl_warmup_start = 250.0 
# kl_warmup_end = 10000.0  
# kl_min = 0.01
# # kl_warmup_start = 250.0 
# # kl_warmup_end = 1000.0
# # kl_min = 0.01
# kl_max = 1.0

#Backup Original Config
###############
batch_size = 128
enc_dim = 128         
con_dim = 128        
ii_dim = 1            
gen_dim = 128         
factors_dim = 32      
var_min = 0.001 
l2reg = 0.00002
ic_prior_var = 0.1 
ar_mean = 0.0       
ar_autocorrelation_tau = 1.0 
ar_noise_variance = 0.1  
num_batches = 2000         
print_every = 100
step_size = 0.05
decay_factor = 0.99
decay_steps = 1 
keep_rate = 0.97 
max_grad_norm = 10.0
kl_warmup_start = 500.0 
kl_warmup_end = 1000.0  
kl_min = 0.01
kl_max = 1.0
##############

lfads_hps = {'data_dim' : data_dim, 'ntimesteps' : ntimesteps,
             'enc_dim' : enc_dim, 'con_dim' : con_dim, 'var_min' : var_min,
             'ic_prior_var' : ic_prior_var, 'ar_mean' : ar_mean,
             'ar_autocorrelation_tau' : ar_autocorrelation_tau,
             'ar_noise_variance' : ar_noise_variance,
             'ii_dim' : ii_dim, 'gen_dim' : gen_dim,
             'factors_dim' : factors_dim,
             'l2reg' : l2reg,
             'batch_size' : batch_size}



lfads_opt_hps = {'num_batches' : num_batches, 'step_size' : step_size,
                 'decay_steps' : decay_steps, 'decay_factor' : decay_factor,
                 'kl_min' : kl_min, 'kl_max' : kl_max, 'kl_warmup_start' : kl_warmup_start,
                 'kl_warmup_end' : kl_warmup_end, 'keep_rate' : keep_rate,
                 'max_grad_norm' : max_grad_norm, 'print_every' : print_every,
                 'adam_b1' : 0.99, 'adam_b2' : 0.999, 'adam_eps' : 1e-1}

assert num_batches >= print_every and num_batches % print_every == 0

# Plot the warmup function and the learning rate decay function.
plt.figure(figsize=(16,4))
plt.subplot(121)
x = onp.arange(0, num_batches, print_every)
kl_warmup_fun = get_kl_warmup_fun(lfads_opt_hps)
plt.plot(x, [kl_warmup_fun(i) for i in onp.arange(1,lfads_opt_hps['num_batches'], print_every)]);
plt.title('KL warmup function')
plt.xlabel('Training batch');

plt.subplot(122)
decay_fun = optimizers.exponential_decay(lfads_opt_hps['step_size'],                                                             
                                         lfads_opt_hps['decay_steps'],                                                           
                                         lfads_opt_hps['decay_factor'])                                                          
plt.plot(x, [decay_fun(i) for i in range(1, lfads_opt_hps['num_batches'], print_every)]);
plt.title('learning rate function')
plt.xlabel('Training batch');                                           

key = random.PRNGKey(onp.random.randint(0, utils.MAX_SEED_INT))
init_params = lfads.lfads_params(key, lfads_hps)


data1 = list(dict.items(lfads_opt_hps))
print(np.shape(data1))

from functools import partial

import jax.numpy as np
from jax import jit, lax, random, vmap
from jax.experimental import optimizers

import lfads_tutorial.distributions as dists
import lfads_tutorial.utils as utils


def sigmoid(x):
  return 0.5 * (np.tanh(x / 2.) + 1)


def linear_params(key, o, u, ifactor=1.0):
  key, skeys = utils.keygen(key, 1)
  ifactor = ifactor / np.sqrt(u)
  return {'w' : random.normal(next(skeys), (o, u)) * ifactor}


def affine_params(key, o, u, ifactor=1.0):
  key, skeys = utils.keygen(key, 1)
  ifactor = ifactor / np.sqrt(u)
  return {'w' : random.normal(next(skeys), (o, u)) * ifactor,
          'b' : np.zeros((o,))}


def gru_params(key, n, u, ifactor=1.0, hfactor=1.0, hscale=0.0):
  key, skeys = utils.keygen(key, 5)
  ifactor = ifactor / np.sqrt(u)
  hfactor = hfactor / np.sqrt(n)

  wRUH = random.normal(next(skeys), (n+n,n)) * hfactor
  wRUX = random.normal(next(skeys), (n+n,u)) * ifactor
  wRUHX = np.concatenate([wRUH, wRUX], axis=1)

  wCH = random.normal(next(skeys), (n,n)) * hfactor
  wCX = random.normal(next(skeys), (n,u)) * ifactor
  wCHX = np.concatenate([wCH, wCX], axis=1)

  return {'h0' : random.normal(next(skeys), (n,)) * hscale,
          'wRUHX' : wRUHX,
          'wCHX' : wCHX,
          'bRU' : np.zeros((n+n,)),
          'bC' : np.zeros((n,))}


def affine(params, x):
  return np.dot(params['w'], x) + params['b']


batch_affine = vmap(affine, in_axes=(None, 0))


def normed_linear(params, x):
  w = params['w']
  w_row_norms = np.sqrt(np.sum(w**2, axis=1, keepdims=True))
  w = w / w_row_norms
  return np.dot(w, x)

batch_normed_linear = vmap(normed_linear, in_axes=(None, 0))


def dropout(x, key, keep_rate):
  do_keep = random.bernoulli(key, keep_rate, x.shape)
  kept_rates = np.where(do_keep, x / keep_rate, 0.0)
  return np.where(keep_rate < 1.0, kept_rates, x)

batch_dropout = vmap(dropout, in_axes=(0, 0, None))


def run_dropout(x_t, key, keep_rate):
  ntime = x_t.shape[0]
  keys = random.split(key, ntime)
  return batch_dropout(x_t, keys, keep_rate)


def gru(params, h, x):
  bfg = 0.5
  hx = np.concatenate([h, x], axis=0)
  ru = np.dot(params['wRUHX'], hx) + params['bRU']
  r, u = np.split(ru, 2, axis=0)
  u = u + bfg
  r = sigmoid(r)
  u = sigmoid(u)
  rhx = np.concatenate([r * h, x])
  c = np.tanh(np.dot(params['wCHX'], rhx) + params['bC'])
  return u * h + (1.0 - u) * c


def make_rnn_for_scan(rnn, params):
  def rnn_for_scan(h, x):
    h = rnn(params, h, x)
    return h, h
  return rnn_for_scan


def run_rnn(rnn_for_scan, x_t, h0):
  _, h_t = lax.scan(rnn_for_scan, h0, x_t)
  return h_t


def run_bidirectional_rnn(params, fwd_rnn, bwd_rnn, x_t):
  fwd_rnn_scan = make_rnn_for_scan(fwd_rnn, params['fwd_rnn'])
  bwd_rnn_scan = make_rnn_for_scan(bwd_rnn, params['bwd_rnn'])

  fwd_enc_t = run_rnn(fwd_rnn_scan, x_t, params['fwd_rnn']['h0'])
  bwd_enc_t = np.flipud(run_rnn(bwd_rnn_scan, np.flipud(x_t),
                                params['bwd_rnn']['h0']))
  full_enc = np.concatenate([fwd_enc_t, bwd_enc_t], axis=1)
  enc_ends = np.concatenate([bwd_enc_t[0], fwd_enc_t[-1]], axis=0)
  return full_enc, enc_ends


def lfads_params(key, lfads_hps):
  key, skeys = utils.keygen(key, 10)

  data_dim = lfads_hps['data_dim']
  ntimesteps = lfads_hps['ntimesteps']
  enc_dim = lfads_hps['enc_dim']
  con_dim = lfads_hps['con_dim']
  ii_dim = lfads_hps['ii_dim']
  gen_dim = lfads_hps['gen_dim']
  factors_dim = lfads_hps['factors_dim']

  ic_enc_params = {'fwd_rnn' : gru_params(next(skeys), enc_dim, data_dim),
                   'bwd_rnn' : gru_params(next(skeys), enc_dim, data_dim)}
  gen_ic_params = affine_params(next(skeys), 2*gen_dim, 2*enc_dim) #m,v <- bi
  ic_prior_params = dists.diagonal_gaussian_params(next(skeys), gen_dim, 0.0,
                                                   lfads_hps['ic_prior_var'])
  con_params = gru_params(next(skeys), con_dim, 2*enc_dim + factors_dim)
  con_out_params = affine_params(next(skeys), 2*ii_dim, con_dim) #m,v
  ii_prior_params = dists.ar1_params(next(skeys), ii_dim,
                                     lfads_hps['ar_mean'],
                                     lfads_hps['ar_autocorrelation_tau'],
                                     lfads_hps['ar_noise_variance'])
  gen_params = gru_params(next(skeys), gen_dim, ii_dim)
  factors_params = linear_params(next(skeys), factors_dim, gen_dim)
  lograte_params = affine_params(next(skeys), data_dim, factors_dim)

  return {'ic_enc' : ic_enc_params,
          'gen_ic' : gen_ic_params, 'ic_prior' : ic_prior_params,
          'con' : con_params, 'con_out' : con_out_params,
          'ii_prior' : ii_prior_params,
          'gen' : gen_params, 'factors' : factors_params,
          'logrates' : lograte_params}


def lfads_encode(params, lfads_hps, key, x_t, keep_rate):
  key, skeys = utils.keygen(key, 3)

  # Encode the input
  x_t = run_dropout(x_t, next(skeys), keep_rate)
  con_ins_t, gen_pre_ics = run_bidirectional_rnn(params['ic_enc'], gru, gru,
                                                 x_t)
  # Push through to posterior mean and variance for initial conditions.
  xenc_t = dropout(con_ins_t, next(skeys), keep_rate)
  gen_pre_ics = dropout(gen_pre_ics, next(skeys), keep_rate)
  ic_gauss_params = affine(params['gen_ic'], gen_pre_ics)
  ic_mean, ic_logvar = np.split(ic_gauss_params, 2, axis=0)
  return ic_mean, ic_logvar, xenc_t


def lfads_decode_one_step(params, lfads_hps, key, keep_rate, c, f, g, xenc):
  keys = random.split(key, 2)
  cin = np.concatenate([xenc, f], axis=0)
  c = gru(params['con'], c, cin)
  cout = affine(params['con_out'], c)
  ii_mean, ii_logvar = np.split(cout, 2, axis=0) # inferred input params
  ii = dists.diag_gaussian_sample(keys[0], ii_mean,
                                  ii_logvar, lfads_hps['var_min'])
  g = gru(params['gen'], g, ii)
  g = dropout(g, keys[1], keep_rate)
  f = normed_linear(params['factors'], g)
  lograte = affine(params['logrates'], f)
  return c, g, f, ii, ii_mean, ii_logvar, lograte
    

def lfads_decode_one_step_scan(params, lfads_hps, keep_rate, state, key_n_xenc):
  key, xenc = key_n_xenc
  c, g, f = state
  state_and_returns = lfads_decode_one_step(params, lfads_hps, key, keep_rate,
                                            c, f, g, xenc)
  c, g, f, ii, ii_mean, ii_logvar, lograte = state_and_returns
  state = (c, g, f)
  return state, state_and_returns


def lfads_decode(params, lfads_hps, key, ic_mean, ic_logvar, xenc_t, keep_rate):
  ntime = lfads_hps['ntimesteps']
  key, skeys = utils.keygen(key, 2)
  c0 = params['con']['h0']
  g0 = dists.diag_gaussian_sample(next(skeys), ic_mean, ic_logvar,
                                  lfads_hps['var_min'])
  f0 = np.zeros((lfads_hps['factors_dim'],))
  T = xenc_t.shape[0]
  keys_t = random.split(next(skeys), T)
  
  state0 = (c0, g0, f0)
  decoder = partial(lfads_decode_one_step_scan, *(params, lfads_hps, keep_rate))
  _, state_and_returns_t = lax.scan(decoder, state0, (keys_t, xenc_t))
  return state_and_returns_t


def lfads(params, lfads_hps, key, x_t, keep_rate):
  key, skeys = utils.keygen(key, 2)

  ic_mean, ic_logvar, xenc_t = \
      lfads_encode(params, lfads_hps, next(skeys), x_t, keep_rate)

  c_t, gen_t, factor_t, ii_t, ii_mean_t, ii_logvar_t, lograte_t = \
      lfads_decode(params, lfads_hps, next(skeys), ic_mean, ic_logvar,
                   xenc_t, keep_rate)
  
  # As this is tutorial code, we're passing everything around.
  return {'xenc_t' : xenc_t, 'ic_mean' : ic_mean, 'ic_logvar' : ic_logvar,
          'ii_t' : ii_t, 'c_t' : c_t, 'ii_mean_t' : ii_mean_t,
          'ii_logvar_t' : ii_logvar_t, 'gen_t' : gen_t, 'factor_t' : factor_t,
          'lograte_t' : lograte_t}


lfads_encode_jit = jit(lfads_encode)
lfads_decode_jit = jit(lfads_decode, static_argnums=(1,))
lfads_jit = jit(lfads, static_argnums=(1,))

batch_lfads = vmap(lfads, in_axes=(None, None, 0, 0, None))


def lfads_losses(params, key, x_bxt,kl_scale, keep_rate,y_bxt):
  lfads_hps = {'data_dim' : 50, 
               'ntimesteps' : 50,
               'enc_dim' : enc_dim, 
               'con_dim' : con_dim, 
               'var_min' : var_min,
               'ic_prior_var' : ic_prior_var, 
               'ar_mean' : ar_mean,
               'ar_autocorrelation_tau' : ar_autocorrelation_tau,
               'ar_noise_variance' : ar_noise_variance,
               'ii_dim' : ii_dim, 
               'gen_dim' : gen_dim,
               'factors_dim' : factors_dim,
               'l2reg' : 2E-05,
               'batch_size' : batch_size}

  B = lfads_hps['batch_size']
  key, skeys = utils.keygen(key, 2)
  keys_b = random.split(next(skeys), B)
  lfads = batch_lfads(params, lfads_hps, keys_b, x_bxt, keep_rate)

  # Sum over time and state dims, average over batch.
  # KL - g0
  ic_post_mean_b = lfads['ic_mean']
  ic_post_logvar_b = lfads['ic_logvar']
  kl_loss_g0_b = dists.batch_kl_gauss_gauss(ic_post_mean_b, ic_post_logvar_b,
                                            params['ic_prior'],
                                            lfads_hps['var_min'])
  kl_loss_g0_prescale = np.sum(kl_loss_g0_b) / B  
  kl_loss_g0 = kl_scale * kl_loss_g0_prescale
  
  # KL - Inferred input
  ii_post_mean_bxt = lfads['ii_mean_t']
  ii_post_var_bxt = lfads['ii_logvar_t']
  keys_b = random.split(next(skeys), B)
  kl_loss_ii_b = dists.batch_kl_gauss_ar1(keys_b, ii_post_mean_bxt,
                                          ii_post_var_bxt, params['ii_prior'],
                                          lfads_hps['var_min'])
  kl_loss_ii_prescale = np.sum(kl_loss_ii_b) / B
  kl_loss_ii = kl_scale * kl_loss_ii_prescale
  
  # Log-likelihood of data given latents.
  lograte_bxt = lfads['lograte_t']
  # log_p_xgz = np.sum(dists.poisson_log_likelihood(x_bxt, lograte_bxt)) / B
  log_p_xgz = np.sum(dists.poisson_log_likelihood(y_bxt, lograte_bxt)) / B

  # L2
  l2reg = lfads_hps['l2reg']
  l2_loss = l2reg * optimizers.l2_norm(params)**2

  loss = -log_p_xgz + kl_loss_g0 + kl_loss_ii + l2_loss
  all_losses = {'total' : loss, 'nlog_p_xgz' : -log_p_xgz,
                'kl_g0' : kl_loss_g0, 'kl_g0_prescale' : kl_loss_g0_prescale,
                'kl_ii' : kl_loss_ii, 'kl_ii_prescale' : kl_loss_ii_prescale,
                'l2' : l2_loss}
  return all_losses


def lfads_training_loss(params, lfads_hps, key, x_bxt, kl_scale, keep_rate,y_bxt):
  losses = lfads_losses(params, key, x_bxt, kl_scale, keep_rate,y_bxt)
  return losses['total']


def posterior_sample_and_average(params, lfads_hps, key, x_txd):
  batch_size = lfads_hps['batch_size']
  skeys = random.split(key, batch_size)  
  x_bxtxd = np.repeat(np.expand_dims(x_txd, axis=0), batch_size, axis=0)
  keep_rate = 1.0
  lfads_dict = batch_lfads(params, lfads_hps, skeys, x_bxtxd, keep_rate)
  return utils.average_lfads_batch(lfads_dict)


batch_lfads_jit = jit(batch_lfads, static_argnums=(1,))
lfads_losses_jit = jit(lfads_losses, static_argnums=(0,))
lfads_training_loss_jit = jit(lfads_training_loss, static_argnums=(1,))
posterior_sample_and_average_jit = jit(posterior_sample_and_average, static_argnums=(1,))




import datetime
import h5py

import jax.numpy as np
from jax import grad, jit, lax, random
from jax.experimental import optimizers

import matplotlib.pyplot as plt
import numpy as onp  # original CPU-backed NumPy
import sklearn

import lfads_tutorial.lfads as lfads
import lfads_tutorial.utils as utils
import numpy as num
import time


def get_kl_warmup_fun(lfads_opt_hps):

  kl_warmup_start = lfads_opt_hps['kl_warmup_start']
  kl_warmup_end = lfads_opt_hps['kl_warmup_end']
  kl_min = lfads_opt_hps['kl_min']
  kl_max = lfads_opt_hps['kl_max']
  def kl_warmup(batch_idx):
    progress_frac = ((batch_idx - kl_warmup_start) /
                     (kl_warmup_end - kl_warmup_start))
    kl_warmup = np.where(batch_idx < kl_warmup_start, kl_min,
                         (kl_max - kl_min) * progress_frac + kl_min)
    return np.where(batch_idx > kl_warmup_end, kl_max, kl_warmup)
  return kl_warmup


def optimize_lfads_core(key, batch_idx_start, num_batches,
                        update_fun, kl_warmup_fun,
                        opt_state):
  lfads_hps = {'data_dim' : 50, 
               'ntimesteps' : 50,
               'enc_dim' : enc_dim, 
               'con_dim' : con_dim, 
               'var_min' : var_min,
               'ic_prior_var' : ic_prior_var, 
               'ar_mean' : ar_mean,
               'ar_autocorrelation_tau' : ar_autocorrelation_tau,
               'ar_noise_variance' : ar_noise_variance,
               'ii_dim' : ii_dim, 
               'gen_dim' : gen_dim,
               'factors_dim' : factors_dim,
               'l2reg' : 2E-05,
               'batch_size' : batch_size}

  lfads_opt_hps = {'num_batches' : num_batches,
                   'step_size' : step_size,
                   'decay_steps' : decay_steps, 
                   'decay_factor' : decay_factor,
                   'kl_min' : kl_min, 
                   'kl_max' : kl_max, 
                   'kl_warmup_start' : kl_warmup_start,
                   'kl_warmup_end' : kl_warmup_end, 
                   'keep_rate' : keep_rate,
                   'max_grad_norm' : max_grad_norm, 
                   'print_every' : print_every,
                   'adam_b1' : 0.9, 
                   'adam_b2' : 0.999, 
                   'adam_eps' : 1e-1}

  key, dkeyg = utils.keygen(key, num_batches) # data
  key, fkeyg = utils.keygen(key, num_batches) # forward pass
  
  def run_update(batch_idx, opt_state):

    train_data, eval_data = utils.split_data(data_bxtxn,train_fraction=train_fraction)
    segmented_train_data, segmented_eval_data = utils.split_data(data_bxtxn,train_fraction=train_fraction)

    print(train_data.shape[0])
    print(lfads_hps['batch_size'])
    print(next(dkeyg))
    didxs = random.randint(next(dkeyg), [lfads_hps['batch_size']], 0,train_data.shape[0])
    kl_warmup = kl_warmup_fun(batch_idx)
    x_bxt = np.array(train_data)[didxs].astype(np.float32)

    segmented_didxs = random.randint(next(dkeyg), [lfads_hps['batch_size']], 0,segmented_train_data.shape[0])
    segmented_kl_warmup = kl_warmup_fun(batch_idx)
    y_bxt = np.array(segmented_train_data)[didxs].astype(np.float32)
    opt_state = update_fun(batch_idx, opt_state, lfads_hps, lfads_opt_hps,next(fkeyg), x_bxt, kl_warmup,y_bxt)
    return opt_state

  lower = batch_idx_start
  upper = batch_idx_start + num_batches
  return lax.fori_loop(lower, upper, run_update, opt_state)


optimize_lfads_core_jit = jit(optimize_lfads_core, static_argnums=(2,3,4,6,7))

def optimize_lfads(key, init_params, lfads_hps, lfads_opt_hps,train_data, eval_data,segmented_train_data,segmented_eval_data):

  all_tlosses = []
  all_elosses = []

  kl_warmup_fun = get_kl_warmup_fun(lfads_opt_hps)
  decay_fun = optimizers.exponential_decay(lfads_opt_hps['step_size'],
                                           lfads_opt_hps['decay_steps'],
                                           lfads_opt_hps['decay_factor'])

  opt_init, opt_update, get_params = optimizers.adam(step_size=decay_fun,
                                                     b1=lfads_opt_hps['adam_b1'],
                                                     b2=lfads_opt_hps['adam_b2'],
                                                     eps=lfads_opt_hps['adam_eps'])
  opt_state = opt_init(init_params)

  def update_w_gc(i, opt_state, lfads_hps, lfads_opt_hps, key, x_bxt,kl_warmup,y_bxt):
    params = get_params(opt_state)
    grads = grad(lfads_training_loss)(params, lfads_hps, key, x_bxt,kl_warmup,lfads_opt_hps['keep_rate'],y_bxt)
    clipped_grads = optimizers.clip_grads(grads, lfads_opt_hps['max_grad_norm'])
    return opt_update(i, clipped_grads, opt_state)

  batch_size = lfads_hps['batch_size']
  num_batches = lfads_opt_hps['num_batches']
  print_every = lfads_opt_hps['print_every']
  num_opt_loops = int(num_batches / print_every)
  params = get_params(opt_state)
  loss_list = []
  for oidx in range(num_opt_loops):
    batch_idx_start = oidx * print_every
    key, tkey, dtkey, dekey = random.split(random.fold_in(key, oidx), 4)
    start_time = time.time()
    opt_state = optimize_lfads_core_jit(tkey, batch_idx_start,print_every, update_w_gc, kl_warmup_fun,opt_state)
    batch_time = time.time() - start_time
    # Losses
    params = get_params(opt_state)
    batch_pidx = batch_idx_start + print_every
    kl_warmup = kl_warmup_fun(batch_idx_start)
    # Training loss
    didxs = onp.random.randint(0, train_data.shape[0], batch_size)
    x_bxt = np.array(train_data)[didxs].astype(np.float32)
    segmented_didxs = onp.random.randint(0, segmented_train_data.shape[0], batch_size)
    y_bxt = np.array(segmented_train_data)[didxs].astype(np.float32)
    tlosses = lfads_losses(params, dtkey, x_bxt,kl_warmup, 1.0,y_bxt)

    # Evaluation loss
    didxs = onp.random.randint(0, eval_data.shape[0], batch_size)
    ex_bxt = np.array (eval_data[didxs].astype(onp.float32))
    segmented_didxs = onp.random.randint(0, segmented_eval_data.shape[0], batch_size)
    yx_bxt = np.array(segmented_eval_data)[didxs].astype(np.float32)
    elosses = lfads_losses(params, dekey, ex_bxt,
                                     kl_warmup, 1.0,y_bxt)
    # Saving, printing.
    all_tlosses.append(tlosses)
    all_elosses.append(elosses)
    loss_list.append(tlosses['total'])
    s1 = "Batches {}-{} in {:0.2f} sec, Step size: {:0.5f}"
    s2 = "    Training losses {:0.0f} = NLL {:0.0f} + KL IC {:0.0f},{:0.0f} + KL II {:0.0f},{:0.0f} + L2 {:0.2f}"
    s3 = "        Eval losses {:0.0f} = NLL {:0.0f} + KL IC {:0.0f},{:0.0f} + KL II {:0.0f},{:0.0f} + L2 {:0.2f}"
    print(s1.format(batch_idx_start+1, batch_pidx, batch_time,
                   decay_fun(batch_pidx)))
    print(s2.format(tlosses['total'], tlosses['nlog_p_xgz'],
                    tlosses['kl_g0_prescale'], tlosses['kl_g0'],
                    tlosses['kl_ii_prescale'], tlosses['kl_ii'],
                    tlosses['l2']))
    print(s3.format(elosses['total'], elosses['nlog_p_xgz'],
                    elosses['kl_g0_prescale'], elosses['kl_g0'],
                    elosses['kl_ii_prescale'], elosses['kl_ii'],
                    elosses['l2']))

    tlosses_thru_training = utils.merge_losses_dicts(all_tlosses)
    elosses_thru_training = utils.merge_losses_dicts(all_elosses)
    optimizer_details = {'tlosses' : tlosses_thru_training,
                         'elosses' : elosses_thru_training}
    
  return params, optimizer_details,loss_list



key = random.PRNGKey(onp.random.randint(0, utils.MAX_SEED_INT))
trained_params, opt_details, listofloss =  optimize_lfads(key, init_params, lfads_hps, lfads_opt_hps,train_data, eval_data,segmented_train_data,segmented_eval_data)
            
from importlib import reload

import json
import numpy as znp

reload(plotting)

znp.save('Small_Dataset_Demo.npy',listofloss)
# znp.save('trained_params.npy', trained_params)

import numpy as onp
import jax.numpy as jnp

factors_w =onp.array(trained_params["factors"]["w"])

ic_prior_mean =onp.array(trained_params["ic_prior"]["mean"])
ic_prior_logvar =onp.array(trained_params["ic_prior"]["logvar"])

ii_prior_mean =onp.array(trained_params["ii_prior"]["mean"])
ii_prior_logatau =onp.array(trained_params["ii_prior"]["logatau"])
ii_prior_lognvar =onp.array(trained_params["ii_prior"]["lognvar"])

gen_h0 =onp.array(trained_params["gen"]["h0"])
gen_wRUHX =onp.array(trained_params["gen"]["wRUHX"])
gen_wCHX =onp.array(trained_params["gen"]["wCHX"])
gen_bRU =onp.array(trained_params["gen"]["bRU"])
gen_bC =onp.array(trained_params["gen"]["bC"])

con_h0 =onp.array(trained_params["con"]["h0"])
con_wRUHX =onp.array(trained_params["con"]["wRUHX"])
con_wCHX =onp.array(trained_params["con"]["wCHX"])
con_bRU =onp.array(trained_params["con"]["bRU"])
con_bC =onp.array(trained_params["con"]["bC"])

gen_ic_w=onp.array(trained_params["gen_ic"]["w"])
gen_ic_b=onp.array(trained_params["gen_ic"]["b"])

con_out_w=onp.array(trained_params["con_out"]["w"])
con_out_b=onp.array(trained_params["con_out"]["b"])

logrates_w=onp.array(trained_params["logrates"]["w"])
logrates_b=onp.array(trained_params["logrates"]["b"])

ic_enc_fwd_rnn_h0=onp.array(trained_params["ic_enc"]["fwd_rnn"]["h0"])
ic_enc_fwd_rnn_wRUHX=onp.array(trained_params["ic_enc"]["fwd_rnn"]["wRUHX"])
ic_enc_fwd_rnn_wCHX=onp.array(trained_params["ic_enc"]["fwd_rnn"]["wCHX"])
ic_enc_fwd_rnn_bRU=onp.array(trained_params["ic_enc"]["fwd_rnn"]["bRU"])
ic_enc_fwd_rnn_bC=onp.array(trained_params["ic_enc"]["fwd_rnn"]["bC"])

ic_enc_bwd_rnn_h0=onp.array(trained_params["ic_enc"]["bwd_rnn"]["h0"])
ic_enc_bwd_rnn_wRUHX=onp.array(trained_params["ic_enc"]["bwd_rnn"]["wRUHX"])
ic_enc_bwd_rnn_wCHX=onp.array(trained_params["ic_enc"]["bwd_rnn"]["wCHX"])
ic_enc_bwd_rnn_bRU=onp.array(trained_params["ic_enc"]["bwd_rnn"]["bRU"])
ic_enc_bwd_rnn_bC=onp.array(trained_params["ic_enc"]["bwd_rnn"]["bC"])

trained_params_numpy1=[
    factors_w,

    ic_prior_mean,
    ic_prior_logvar,

    ii_prior_mean,
    ii_prior_logatau,
    ii_prior_lognvar,

    gen_h0,
    gen_wRUHX,
    gen_wCHX,
    gen_bRU,
    gen_bC,

    con_h0,
    con_wRUHX,
    con_wCHX,
    con_bRU,
    con_bC,

    gen_ic_w,
    gen_ic_b,

    con_out_w,
    con_out_b,

    logrates_w,
    logrates_b,

    ic_enc_fwd_rnn_h0,
    ic_enc_fwd_rnn_wRUHX,
    ic_enc_fwd_rnn_wCHX,
    ic_enc_fwd_rnn_bRU,
    ic_enc_fwd_rnn_bC,
    ic_enc_bwd_rnn_h0,
    ic_enc_bwd_rnn_wRUHX,
    ic_enc_bwd_rnn_wCHX,
    ic_enc_bwd_rnn_bRU,
    ic_enc_bwd_rnn_bC
]

import json
import numpy as np

np.save('Small_Dataset_Demo_Weights.npy', trained_params_numpy1)

# def plot_rescale_fun(a): 
#     fac = max_firing_rate * data_dt
#     return renormed_fun(a) * fac


# bidx = my_example_bidx - eval_data_offset
# bidx = 0


# def posterior_sample_and_average(params, lfads_hps, key, x_txd):
#   batch_size = lfads_hps['batch_size']
#   skeys = random.split(key, batch_size)  
#   x_bxtxd = np.repeat(np.expand_dims(x_txd, axis=0), batch_size, axis=0)
#   keep_rate = 1.0
#   lfads_dict = batch_lfads(params, lfads_hps, skeys, x_bxtxd, keep_rate)
#   return utils.average_lfads_batch(lfads_dict)


# nexamples_to_save = 1
# for eidx in range(nexamples_to_save):
#     fkey = random.fold_in(key, eidx)
#     psa_example = eval_data[bidx,:,:].astype(np.float32)
#     psa_dict = lfads.posterior_sample_and_average(trained_params_numpy1, lfads_hps, fkey, psa_example)
#     plotting.plot_lfads(psa_example, psa_dict)
from importlib import reload

reload(plotting)
#reload(lfads)

def plot_rescale_fun(a): 
    fac = max_firing_rate * data_dt
    return renormed_fun(a) * fac

def posterior_sample_and_average(params, lfads_hps, key, x_txd):
  batch_size = lfads_hps['batch_size']
  skeys = random.split(key, batch_size)  
  x_bxtxd = np.repeat(np.expand_dims(x_txd, axis=0), batch_size, axis=0)
  keep_rate = 1.0
  lfads_dict = batch_lfads(params, lfads_hps, skeys, x_bxtxd, keep_rate)
  return utils.average_lfads_batch(lfads_dict)

bidx = my_example_bidx - eval_data_offset
bidx = 0

nexamples_to_save = 10
for eidx in range(nexamples_to_save):
    fkey = random.fold_in(key, eidx)
    psa_example = eval_data[bidx,:,:].astype(np.float32)
    psa_dict = lfads.posterior_sample_and_average(trained_params_numpy1, lfads_hps, fkey, psa_example)
    plotting.plot_lfads(psa_example, psa_dict, None, eval_data_offset+bidx, plot_rescale_fun)