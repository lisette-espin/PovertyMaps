# https://gist.github.com/eaed885b4b56586ad82c952c801fbc00.git
# https://gist.dreamtobe.cn/innat/eaed885b4b56586ad82c952c801fbc00

import numpy as np
import random, math
from scipy.stats import beta

def binarise_mask(mask, lam, in_shape, max_soft=0.0):
  """ Binarises a given low frequency image such that it has mean lambda.
  :param mask: Low frequency image, usually the result of `make_low_freq_image`
  :param lam: Mean value of final mask
  :param in_shape: Shape of inputs
  :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
  :return:
  """
  idx = mask.reshape(-1).argsort()[::-1]
  mask = mask.reshape(-1)
  num = math.ceil(lam * mask.size) if random.random() > 0.5 else math.floor(lam * mask.size)

  eff_soft = max_soft
  if max_soft > lam or max_soft > (1-lam):
    eff_soft = min(lam, 1-lam)

  soft = int(mask.size * eff_soft)
  num_low = num - soft
  num_high = num + soft

  mask[idx[:num_high]] = 1
  mask[idx[num_low:]] = 0
  mask[idx[num_low:num_high]] = np.linspace(1, 0, (num_high - num_low))

  mask = mask.reshape((1, *in_shape))
  return mask

def make_low_freq_image(decay, shape, ch=1):
  """ Sample a low frequency image from fourier space
  :param decay_power: Decay power for frequency decay prop 1/f**d
  :param shape: Shape of desired mask, list up to 3 dims
  :param ch: Number of channels for desired mask
  """
  freqs = fftfreqnd(*shape)
  spectrum = get_spectrum(freqs, decay, ch, *shape)#.reshape((1, *shape[:-1], -1))
  spectrum = spectrum[:, 0] + 1j * spectrum[:, 1]
  mask = np.real(np.fft.irfftn(spectrum, shape))

  if len(shape) == 1:
    mask = mask[:1, :shape[0]]
  if len(shape) == 2:
    mask = mask[:1, :shape[0], :shape[1]]
  if len(shape) == 3:
    mask = mask[:1, :shape[0], :shape[1], :shape[2]]

  mask = mask
  mask = (mask - mask.min())
  mask = mask / mask.max()
  return mask


def sample_lam(alpha, reformulate=False):
  """ Sample a lambda from symmetric beta distribution with given alpha
  :param alpha: Alpha value for beta distribution
  :param reformulate: If True, uses the reformulation of [1].
  """
  if reformulate:
    lam = beta.rvs(alpha+1, alpha)
  else:
    lam = beta.rvs(alpha, alpha)

  return lam

def sample_mask(alpha, decay_power, shape, max_soft=0.0, reformulate=False):
  """ Samples a mean lambda from beta distribution parametrised by alpha, creates a low frequency image and binarises
  it based on this lambda
  :param alpha: Alpha value for beta distribution from which to sample mean of mask
  :param decay_power: Decay power for frequency decay prop 1/f**d
  :param shape: Shape of desired mask, list up to 3 dims
  :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
  :param reformulate: If True, uses the reformulation of [1].
  """
  if isinstance(shape, int):
    shape = (shape,)

  # Choose lambda
  lam = sample_lam(alpha, reformulate)

  # Make mask, get mean / std
  mask = make_low_freq_image(decay_power, shape)
  mask = binarise_mask(mask, lam, shape, max_soft)

  return lam, mask


def fftfreqnd(h, w=None, z=None):
  """ Get bin values for discrete fourier transform of size (h, w, z)
  :param h: Required, first dimension size
  :param w: Optional, second dimension size
  :param z: Optional, third dimension size
  """
  fz = fx = 0
  fy = np.fft.fftfreq(h)

  if w is not None:
    fy = np.expand_dims(fy, -1)

    if w % 2 == 1:
      fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
      fx = np.fft.fftfreq(w)[: w // 2 + 1]

  if z is not None:
    fy = np.expand_dims(fy, -1)
    if z % 2 == 1:
      fz = np.fft.fftfreq(z)[:, None]
    else:
      fz = np.fft.fftfreq(z)[:, None]

  return np.sqrt(fx * fx + fy * fy + fz * fz)

def get_spectrum(freqs, decay_power, ch, h, w=0, z=0):
  """ Samples a fourier image with given size and frequencies decayed by decay power
  :param freqs: Bin values for the discrete fourier transform
  :param decay_power: Decay power for frequency decay prop 1/f**d
  :param ch: Number of channels for the resulting mask
  :param h: Required, first dimension size
  :param w: Optional, second dimension size
  :param z: Optional, third dimension size
  """
  scale = np.ones(1) / (np.maximum(freqs, np.array([1. / max(w, h, z)])) ** decay_power)

  param_size = [ch] + list(freqs.shape) + [2]
  param = np.random.randn(*param_size)

  scale = np.expand_dims(scale, -1)[None, :]

  return scale * param