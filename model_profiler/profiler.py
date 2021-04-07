# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 17:45:33 2021

@author: Talha
"""

from utils import get_available_gpus, get_param, keras_model_memory_usage, count_flops, mem_for_storing_weights
#%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import EfficientNetB4, EfficientNetB7
from tabulate import tabulate

Batch_size = 1
model = EfficientNetB4(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax")

#model.save_weights('ef4.h5')
# model.fit(data, one_hot_labels, epochs=Epoch, batch_size=Batch_size,
#         verbose=1)
#%%
use_units = ['GPU IDs', 'BFLOPs', 'GB', 'Million', 'MB']



def model_profiler(model, Batch_size, use_units):
    
    gpus = get_available_gpus()
    flops = count_flops(use_units[1], model, Batch_size)
    mem = keras_model_memory_usage(use_units[2], model, Batch_size)
    param = get_param(use_units[3], model)
    mem_req = mem_for_storing_weights(use_units[4], model)
    
    values = [gpus, flops, mem, param, mem_req]
    return values

values = model_profiler(model, Batch_size, use_units)

Profile = ['Selected GPUs', 'No. of FLOPs', 'GPU Memory Requirement',
           'Model Parameters', 'Memory Required by Model Weights']

full_profile = np.concatenate((
                                np.asarray(Profile).reshape(-1,1),
                                np.asarray(values).reshape(-1,1),
                                np.asarray(use_units).reshape(-1,1)
                                )
                            , 1)
print(tabulate(
                np.ndarray.tolist(full_profile),
                headers = ["Model Profile", "Value", "Unit"],
                tablefmt="github"
                )
    )
