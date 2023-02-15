################################################################################
# Dependencies
################################################################################

import sys
from tensorflow.keras.models import Sequential 
from tensorflow.keras.backend import set_image_data_format 
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization 
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, ActivityRegularization
from tensorflow.keras import optimizers, utils 
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
#from tensorflow.keras.backend import clear_session
from keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.regularizers import l1_l2

from utils.constants import *
from utils.augmentation import RandomColorDistortion #CutMix, MixUp, MosaicMix, FMix, 

################################################################################
# Functions
################################################################################

def load_existing_model(fn):
  K.clear_session()
  model = load_model(fn, custom_objects={'RandomColorDistortion':RandomColorDistortion()}) 
  # 'R2':R2(name='R2'), # r2 does not work when predictig both mean and std
  #model.summary()
  return model

def define_model(params):
  # model_name, nclasses, pixels, bands, lr, dropout, optimizer_name, **kwargs):
  #  rotation=0.2, contrast=0.9, translation=0.2, zoom=0.6,
  #  contrast_range=[0.5, 1.5], brightness_delta=[-0.2, 0.2]):

  if params['model_name'] == 'aug_cnn_mp_dp_relu_sigmoid_adam':
    fnc = model_aug_cnn_mp_dp_relu_sigmoid_adam
  elif params['model_name'] == 'aug_cnn_mp_dp_relu_sigmoid_adam_mean_regression':
    fnc = model_aug_cnn_mp_dp_relu_sigmoid_adam_mean_regression
  elif params['model_name'] == 'aug_cnn_mp_dp_relu_sigmoid_adam_mean_std_regression':
    fnc = model_aug_cnn_mp_dp_relu_sigmoid_adam_mean_std_regression
  elif params['model_name'] == 'cnn_mp_dp_relu_sigmoid_adam_mean_std_regression':
    fnc = model_cnn_mp_dp_relu_sigmoid_adam_mean_std_regression
  else:
    raise Exception("model name does not exist.")
  
  return fnc(params) #rotation, contrast, translation, zoom, contrast_range, brightness_delta)
  
  
def _get_optimizer(optimizer_name):
  if optimizer_name.lower() == 'adam':
    return Adam
    
  if optimizer_name.lower() == 'sgd':
    return SGD
  
  if optimizer_name.lower() == 'rmsprop':
    return RMSprop

  raise Exception("Optimizer is not implemented.")

def validate_online_augmentation_params(params):
  ks = ['rotation','contrast','translation','zoom','contrast_range','brightness_delta']
  for k in ks:
    if k not in params:
      print('hyper-params:',params)
      raise Exception("Online augmentation hyper-parameters are missing")


def validate_model_params(params):
  ks = ['pixels','bands','dropout','n_classes','lr','optimizer_name']
  for k in ks:
    if k not in params:
      print('hyper-params:',params)
      raise Exception("Model hyper-parameters are missing")

def model_aug_cnn_mp_dp_relu_sigmoid_adam(params):
  # https://medium.datadriveninvestor.com/patch-based-cover-type-classification-using-satellite-imagery-a67edeae7e24

  validate_online_augmentation_params(params)
  validate_model_params(params)

  model = Sequential() 
  
  model.add(layers.experimental.preprocessing.Rescaling(1./255., input_shape=(params['pixels'], params['pixels'], params['bands'])))
  model.add(layers.experimental.preprocessing.RandomFlip("horizontal"))
  model.add(layers.experimental.preprocessing.RandomFlip("vertical"))
  model.add(layers.experimental.preprocessing.RandomRotation(params['rotation']))
  model.add(layers.experimental.preprocessing.RandomContrast(params['contrast']))
  model.add(layers.experimental.preprocessing.RandomTranslation(height_factor=params['translation'], width_factor=params['translation']))
  model.add(layers.experimental.preprocessing.RandomZoom(height_factor=params['zoom'], width_factor=params['zoom']))
  model.add(RandomColorDistortion(name='coldist_1', contrast_range=params['contrast_range'], brightness_delta=params['brightness_delta']))

  model.add(Conv2D(28, (3, 3), padding='same'))
  model.add(Activation('relu')) 

  model.add(Conv2D(28, (3, 3), padding='same')) 
  model.add(Activation('relu')) 
  model.add(MaxPool2D(2,2)) 

  model.add(Conv2D(56, (3, 3),padding='same')) 
  model.add(Activation('relu')) 

  model.add(Conv2D(56, (3, 3), padding='same')) 
  model.add(Activation('relu')) 
  model.add(MaxPool2D(2,2)) 

  model.add(Conv2D(112, (3, 3), padding='same')) 
  model.add(Activation('relu')) 

  model.add(Conv2D(112, (3, 3), padding='same')) 
  model.add(Activation('relu')) 
  model.add(MaxPool2D(2,2)) 

  model.add(Flatten()) 
  model.add(Dense(784)) 
  model.add(Activation('relu')) 
  model.add(Dropout(params['dropout'])) 
  
  # classification
  model.add(Dense(params['n_classes'])) 
  model.add(Activation('sigmoid')) 

  opt = _get_optimizer(params['optimizer_name'])(learning_rate=params['lr'])
  model.compile(optimizer=opt, loss=LOSS_CLASSIFICATION, metrics=METRICS_CLASSIFICATION)
  return model

def model_aug_cnn_mp_dp_relu_sigmoid_adam_mean_regression(params):
  # https://medium.datadriveninvestor.com/patch-based-cover-type-classification-using-satellite-imagery-a67edeae7e24

  validate_online_augmentation_params(params)
  validate_model_params(params)

  model = Sequential() 

  model.add(layers.experimental.preprocessing.Rescaling(1./255., input_shape=(params['pixels'], params['pixels'], params['bands'])))
  model.add(layers.experimental.preprocessing.RandomFlip("horizontal"))
  model.add(layers.experimental.preprocessing.RandomFlip("vertical"))
  model.add(layers.experimental.preprocessing.RandomRotation(params['rotation']))
  model.add(layers.experimental.preprocessing.RandomContrast(params['contrast']))
  model.add(layers.experimental.preprocessing.RandomTranslation(height_factor=params['translation'], width_factor=params['translation']))
  model.add(layers.experimental.preprocessing.RandomZoom(height_factor=params['zoom'], width_factor=params['zoom']))
  model.add(RandomColorDistortion(name='coldist_1', contrast_range=params['contrast_range'], brightness_delta=params['brightness_delta']))

  model.add(Conv2D(28, (3, 3), padding='same'))
  model.add(Activation('relu')) 

  model.add(Conv2D(28, (3, 3), padding='same')) 
  model.add(Activation('relu')) 
  model.add(MaxPool2D(2,2)) 

  model.add(Conv2D(56, (3, 3),padding='same')) 
  model.add(Activation('relu')) 

  model.add(Conv2D(56, (3, 3), padding='same')) 
  model.add(Activation('relu')) 
  model.add(MaxPool2D(2,2)) 

  model.add(Conv2D(112, (3, 3), padding='same')) 
  model.add(Activation('relu')) 

  model.add(Conv2D(112, (3, 3), padding='same')) 
  model.add(Activation('relu')) 
  model.add(MaxPool2D(2,2)) 

  model.add(Flatten()) 
  model.add(Dense(784)) 
  model.add(Activation('relu')) 
  model.add(Dropout(params['dropout']))
  
  # regression
  model.add(Dense(1)) # mean iwi
  model.add(Activation('linear')) 

  opt = _get_optimizer(params['optimizer_name'])(learning_rate=params['lr'])
  model.compile(optimizer=opt, loss=LOSS_REGRESSION, metrics=METRICS_REGRESSION)
  return model


def model_aug_cnn_mp_dp_relu_sigmoid_adam_mean_std_regression(params):
  # https://medium.datadriveninvestor.com/patch-based-cover-type-classification-using-satellite-imagery-a67edeae7e24

  validate_online_augmentation_params(params)
  validate_model_params(params)

  model = Sequential() 

  model.add(layers.experimental.preprocessing.Rescaling(1./255., input_shape=(params['pixels'], params['pixels'], params['bands'])))
  model.add(layers.experimental.preprocessing.RandomFlip("horizontal"))
  model.add(layers.experimental.preprocessing.RandomFlip("vertical"))
  model.add(layers.experimental.preprocessing.RandomRotation(params['rotation']))
  model.add(layers.experimental.preprocessing.RandomContrast(params['contrast']))
  model.add(layers.experimental.preprocessing.RandomTranslation(height_factor=params['translation'], width_factor=params['translation']))
  model.add(layers.experimental.preprocessing.RandomZoom(height_factor=params['zoom'], width_factor=params['zoom']))
  model.add(RandomColorDistortion(name='coldist_1', contrast_range=params['contrast_range'], brightness_delta=params['brightness_delta']))

  model.add(Conv2D(28, (3, 3), padding='same'))
  model.add(Activation('relu')) 

  model.add(Conv2D(28, (3, 3), padding='same')) 
  model.add(Activation('relu')) 
  model.add(MaxPool2D(2,2)) 

  model.add(Conv2D(56, (3, 3),padding='same')) 
  model.add(Activation('relu')) 

  model.add(Conv2D(56, (3, 3), padding='same')) 
  model.add(Activation('relu')) 
  model.add(MaxPool2D(2,2)) 

  model.add(Conv2D(112, (3, 3), padding='same')) 
  model.add(Activation('relu')) 

  model.add(Conv2D(112, (3, 3), padding='same')) 
  model.add(Activation('relu')) 
  model.add(MaxPool2D(2,2)) 

  model.add(Flatten()) 
  model.add(Dense(784)) 
  model.add(Activation('relu')) 
  model.add(Dropout(params['dropout']))
  
  # regression
  model.add(Dense(2)) # mean, std iwi
  model.add(Activation('linear')) 

  opt = _get_optimizer(params['optimizer_name'])(learning_rate=params['lr'])
  model.compile(optimizer=opt, loss=LOSS_REGRESSION, metrics=METRICS_REGRESSION)
  return model


def model_cnn_mp_dp_relu_sigmoid_adam_mean_std_regression(params):
  # https://medium.datadriveninvestor.com/patch-based-cover-type-classification-using-satellite-imagery-a67edeae7e24

  validate_model_params(params)

  model = Sequential() 

  model.add(layers.experimental.preprocessing.Rescaling(1./255., input_shape=(params['pixels'], params['pixels'], params['bands'])))
  
  model.add(Conv2D(28, (3, 3), padding='same'))
  model.add(Activation('relu')) 

  model.add(Conv2D(28, (3, 3), padding='same')) 
  model.add(Activation('relu')) 
  model.add(MaxPool2D(2,2)) 

  model.add(Conv2D(56, (3, 3),padding='same')) 
  model.add(Activation('relu')) 

  model.add(Conv2D(56, (3, 3), padding='same')) 
  model.add(Activation('relu')) 
  model.add(MaxPool2D(2,2)) 

  model.add(Conv2D(112, (3, 3), padding='same')) 
  model.add(Activation('relu')) 

  model.add(Conv2D(112, (3, 3), padding='same')) 
  model.add(Activation('relu')) 
  model.add(MaxPool2D(2,2)) 

  model.add(Flatten()) 
  model.add(Dense(784)) 
  model.add(Activation('relu')) 
  model.add(Dropout(params['dropout']))
  
  # regression
  model.add(Dense(2)) # mean, std
  model.add(Activation('linear')) 

  opt = _get_optimizer(params['optimizer_name'])(learning_rate=params['lr'])
  model.compile(optimizer=opt, loss=LOSS_REGRESSION, metrics=METRICS_REGRESSION)
  return model
