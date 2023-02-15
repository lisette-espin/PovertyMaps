import os
import time

################################################################
# Functions:
################################################################

def google_cloud_shutdown(python_script):
    print('Getting ready to shutdown (wait 5min)...')
    time.sleep(60*5)
    
    command = "python {}".format(python_script)
    print("running: {}".format(command))
    os.system(command)

def check_gpu():
  import tensorflow as tf
  import tensorflow_hub as hub
  import subprocess
  from keras import backend as K
  
  print("===================================================")
  print("TF version:", tf.__version__)
  print("Hub version:", hub.__version__)
  print(tf.test.is_built_with_cuda())
  print(tf.config.list_physical_devices())
  print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
  print("===================================================")        
  os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
  print(os.environ["TF_GPU_ALLOCATOR"])
  print("===================================================")   
  K.clear_session()
  print('clear session')
  print("===================================================") 
  
  if len(tf.config.list_physical_devices('GPU'))==0:
    print("There is no GPU activated.")
  else:
    subprocess.call(['nvidia-smi']) 
    print("GPU is ready!")
