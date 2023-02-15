# https://www.kaggle.com/ipythonx/tf-keras-complex-augmentation-in-data-generator?scriptVersionId=72111687
import numpy as np
import pandas as pd
from glob import glob
import albumentations as A 
import os, warnings, random

# sklearn
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# tf 
import tensorflow as tf
from tensorflow.keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.simplefilter('ignore')

#########################################################
# SequenceGenerator
#########################################################
class SequenceGenerator(tf.keras.utils.Sequence):
  def __init__(self, X, y, num_classes,
               batch_size, dim, shuffle=True,
               use_mixup=False, use_cutmix=False,
               use_fmix=False, use_mosaicmix=False, transform=None):
    self.dim  = dim
    self.num_classes = num_classes
    self.X = X
    self.y = y
    self.shuffle  = shuffle
    self.use_cutmix = use_cutmix
    self.use_mixup  = use_mixup
    self.use_fmix   = use_fmix 
    self.use_mosaicmix = use_mosaicmix
    self.batch_size = batch_size
    self.augment = transform
    self.list_idx   = np.arange(self.X.shape[0])
    self.on_epoch_end()

  def __len__(self):
    return int(np.ceil(float(self.X.shape[0]) / float(self.batch_size)))
    
  def __getitem__(self, index):
    batch_idx = self.indices[index*self.batch_size:(index+1)*self.batch_size]
    idx = [self.list_idx[k] for k in batch_idx]

    Data   = np.empty((self.batch_size, *self.dim))
    Target = np.empty((self.batch_size, self.num_classes), dtype = np.float32)

    for i, k in enumerate(idx):
      # load the image
      image = self.X[k,:,:,:]
      res = self.augment(image=image)
      image = res['image']

      # assign 
      Data[i,] =  image
      Target[i,] = self.y[k,]

    # cutmix 
    if self.use_cutmix:
      Data, Target = CutMix(Data, Target, self.num_classes, self.dim[0])

    # mixup 
    if self.use_mixup:
      Data, Target = MixUp(Data, Target, self.num_classes, self.dim[0]) 

    # fmix 
    if self.use_fmix:
      Data, Target = FMix(Data, Target, self.dim[0])

    if self.use_mosaicmix:
      Data, Target = MosaicMix(Data, Target, self.dim[0]) 

    return Data, Target 
    
  def on_epoch_end(self):
    self.indices = np.arange(len(self.list_idx))
    if self.shuffle:
      np.random.shuffle(self.indices)


#########################################################
# CutMix
#########################################################         
def CutMix(image, label, NUM_CLASSES, DIM, PROBABILITY = 1.0):
  # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
  # output - a batch of images with cutmix applied

  imgs = []; labs = []
  for j in range(len(image)):
    # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
    P = tf.cast( tf.random.uniform([],0,1)<=PROBABILITY, tf.int32)

    # CHOOSE RANDOM IMAGE TO CUTMIX WITH
    k = tf.cast( tf.random.uniform([],0,len(image)),tf.int32)

    # CHOOSE RANDOM LOCATION
    x = tf.cast( tf.random.uniform([],0,DIM),tf.int32)
    y = tf.cast( tf.random.uniform([],0,DIM),tf.int32)

    b = tf.random.uniform([],0,1) # this is beta dist with alpha=1.0

    WIDTH = tf.cast( DIM * tf.math.sqrt(1-b),tf.int32) * P
    ya = tf.math.maximum(0,y-WIDTH//2)
    yb = tf.math.minimum(DIM,y+WIDTH//2)
    xa = tf.math.maximum(0,x-WIDTH//2)
    xb = tf.math.minimum(DIM,x+WIDTH//2)

    # MAKE CUTMIX IMAGE
    one = image[j,ya:yb,0:xa,:]
    two = image[k,ya:yb,xa:xb,:]
    three = image[j,ya:yb,xb:DIM,:]
    middle = tf.concat([one,two,three],axis=1)
    img = tf.concat([image[j,0:ya,:,:],middle,image[j,yb:DIM,:,:]],axis=0)
    imgs.append(img)

    # MAKE CUTMIX LABEL
    a = tf.cast(WIDTH*WIDTH/DIM/DIM,tf.float32)
    labs.append((1-a)*label[j] + a*label[k])

  # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
  image2 = tf.reshape(tf.stack(imgs),(len(image),DIM,DIM,3))
  label2 = tf.reshape(tf.stack(labs),(len(image),NUM_CLASSES))

  return image2,label2
  
#########################################################
# MixUp
#########################################################
def MixUp(image, label, NUM_CLASSES, DIM, PROBABILITY = 1.0):
  # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
  # output - a batch of images with mixup applied

  imgs = []; labs = []
  for j in range(len(image)):
    # DO MIXUP WITH PROBABILITY DEFINED ABOVE
    P = tf.cast( tf.random.uniform([],0,1)<=PROBABILITY, tf.float32)

    # CHOOSE RANDOM
    k = tf.cast( tf.random.uniform([],0,len(image)),tf.int32)
    a = tf.random.uniform([],0,1)*P # this is beta dist with alpha=1.0

    # MAKE MIXUP IMAGE
    img1 = image[j,]
    img2 = image[k,]
    imgs.append((1-a)*img1 + a*img2)

    # MAKE CUTMIX LABEL
    labs.append((1-a)*label[j] + a*label[k])

  # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
  image2 = tf.reshape(tf.stack(imgs),(len(image),DIM,DIM,3))
  label2 = tf.reshape(tf.stack(labs),(len(image),NUM_CLASSES))
  return image2,label2
  
#########################################################
# MosaicMix
#########################################################
def MosaicMix(image, label, DIM, minfrac=0.25, maxfrac=0.75):
  xc, yc  = np.random.randint(DIM * minfrac, DIM * maxfrac, (2,))
  indices = np.random.permutation(int(image.shape[0]))
  mosaic_image = np.zeros((DIM, DIM, 3), dtype=np.float32)
  final_imgs   = []

  # Iterate over the full indices 
  for j in range(len(indices)): 
    # Take 4 sample for to create a mosaic sample randomly 
    rand4indices = [j] + random.sample(list(indices), 3) 

    # Make mosaic with 4 samples 
    for i in range(len(rand4indices)):
      if i == 0:    # top left
        x1a, y1a, x2a, y2a =  0,  0, xc, yc
        x1b, y1b, x2b, y2b = DIM - xc, DIM - yc, DIM, DIM # from bottom right        
      elif i == 1:  # top right
        x1a, y1a, x2a, y2a = xc, 0, DIM , yc
        x1b, y1b, x2b, y2b = 0, DIM - yc, DIM - xc, DIM # from bottom left
      elif i == 2:  # bottom left
        x1a, y1a, x2a, y2a = 0, yc, xc, DIM
        x1b, y1b, x2b, y2b = DIM - xc, 0, DIM, DIM-yc   # from top right
      elif i == 3:  # bottom right
        x1a, y1a, x2a, y2a = xc, yc,  DIM, DIM
        x1b, y1b, x2b, y2b = 0, 0, DIM-xc, DIM-yc    # from top left

      # Copy-Paste
      mosaic_image[y1a:y2a, x1a:x2a] = image[i,][y1b:y2b, x1b:x2b]

    # Append the Mosiac samples
    final_imgs.append(mosaic_image)

  return final_imgs, label

#########################################################
# FMix
#########################################################
from utils.fmix_utils import sample_mask
def FMix(image, label, DIM,  alpha=1, decay_power=3, max_soft=0.0, reformulate=False):
  lam, mask = sample_mask(alpha, decay_power,(DIM, DIM), max_soft, reformulate)
  index = tf.constant(np.random.permutation(int(image.shape[0])))
  mask  = np.expand_dims(mask, -1)

  # samples 
  image1 = image * mask
  image2 = tf.gather(image, index) * (1 - mask)
  image3 = image1 + image2

  # labels
  label1 = label * lam 
  label2 = tf.gather(label, index) * (1 - lam)
  label3 = label1 + label2 
  return image3, label3

#########################################################
# Transforms
#########################################################
# For Training 
def albu_transforms_train(data_resize): 
  return A.Compose([
          A.ToFloat(),
          A.Resize(data_resize, data_resize),
      ], p=1.)

# For Validation 
def albu_transforms_valid(data_resize): 
  return A.Compose([
          A.ToFloat(),
          A.Resize(data_resize, data_resize),
      ], p=1.)