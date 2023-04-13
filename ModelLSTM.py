#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 23:19:10 2022

@author: McGillDocs
"""

# Tensorflow

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import LSTM, Bidirectional, BatchNormalization, Reshape


# ------------------------------------------- # 
# Model functions
# ------------------------------------------- # 

def LSTMModel(input_shape):
    """
    LSTM based model

    Parameters
    ----------
    input_shape : TYPE
        DESCRIPTION.

    Returns
    -------
    model : TYPE
        DESCRIPTION.

    """
    
    X_input = Input(shape = input_shape)

    # Layer 1: CONV layer
    # Add a Conv1D with 196 units, kernel size of 15 and stride of 4
    
    X = Conv1D(filters=32,kernel_size=12,strides=2)(X_input)
    # Batch normalization
    X = BatchNormalization()(X_input)
    # ReLu activation
    X = Activation('relu')(X)
    # dropout (use 0.8)
    X = Dropout(rate=0.1)(X)                                  
    
    # Layer 2: First GRU Layer
    # GRU (use 128 units and return the sequences)
    X = LSTM(units=32, return_sequences = True)(X_input)
    # dropout (use 0.8)
    X = Dropout(rate=0.1)(X)
    # Batch normalization.
    X = BatchNormalization()(X)                           
    
    # Layer 3: Second GRU Layer
    # GRU (use 128 units and return the sequences)
    X = LSTM(units=32, return_sequences = True)(X)
    # dropout (use 0.8)
    X = Dropout(rate=0.1)(X)     
    # Batch normalization
    X = BatchNormalization()(X)
    # dropout (use 0.8)
    X = Dropout(rate=0.1)(X)                 
    
    # Step 4: Time-distributed dense layer (≈1 line)
    # TimeDistributed  with sigmoid activation 
    X = TimeDistributed(Dense(50, activation = "relu"))(X) 
    X = TimeDistributed(Dense(1, activation = "relu"))(X) 

    ### END CODE HERE ###

    model = Model(inputs = X_input, outputs = X)
    
    return model  