#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 20:06:44 2022

@author: McGillDocs
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from ModelGRU import GruModel
from ModelLSTM import LSTMModel



def TrainGRUModel(x_train, y_train, x_dev, y_dev):

    model = GruModel(input_shape = x_train[1, :, :].shape)
    
    # model = tf.keras.models.load_model('Models/GRU_Model.h5')
    
    model.summary()

    opt = Adam(lr=1e-1, beta_1=0.9, beta_2=0.999)
    
    model.compile(loss = tf.keras.losses.MeanSquaredError(),
                  optimizer = opt,
                  metrics = tf.keras.metrics.RootMeanSquaredError())
    
    model.fit(x_train, y_train, batch_size = 100, epochs=1)
    
    model.save('Models/GRU_Model.h5')
    
    model.evaluate(x_dev, y_dev)
    
    return model


def TrainLSTMModel(x_train, y_train, x_dev, y_dev):

    model = LSTMModel(input_shape = x_train[1, :, :].shape)
    
    # model = tf.keras.models.load_model('Models/LSTM_Model.h5')
    
    model.summary()

    opt = Adam(lr=0.5, beta_1=0.9, beta_2=0.999)
    
    model.compile(loss = tf.keras.losses.MeanSquaredError(),
                  optimizer = opt,
                  metrics = tf.keras.metrics.RootMeanSquaredError())
    
    model.fit(x_train, y_train, batch_size = 100, epochs=1)
    
    model.save('Models/LSTM_Model.h5')
    
    model.evaluate(x_dev, y_dev)
    
    return model
    