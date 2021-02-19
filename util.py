# Collection of utility functions.

import tensorflow as tf 
import numpy as np

def run_tf_check():
    print("Current tensorflow version =", tf.__version__)
    print("Num GPUs Available: ", 
        len(tf.config.experimental.list_physical_devices('GPU')))
    print("Is built with CUDA: ", tf.test.is_built_with_cuda())   
    print("\nList of detected devices and attributes:\n", 
        tf.python.client.device_lib.list_local_devices())

