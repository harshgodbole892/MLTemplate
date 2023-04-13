#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 22:38:21 2022

@author: McGillDocs

"""

import numpy as np
import tensorflow as tf
import pandas as pd
import Preprocessing
from TrainModelFunctional import TrainLSTMModel as tmf

# ------------------------------------------- # 
# Options:
# ------------------------------------------- # 

options = {
    'load_data' : True,
    'preprocessing' : True,
    'train_model' : True,
    
    # Preprocessing Options
    'pad_x_data' : True ,
    'one_hot_features' : True,
    
    # Model Options
}
        

# ------------------------------------------- # 
# Loading Data
# ------------------------------------------- # 

if options['load_data']:
        
    # Make numpy values easier to read.
    np.set_printoptions(precision=3, suppress=True)
    
    y_out_label = 'tm'
    
    # Import data as pandas data frame
    print("Phase 01:\nLoading Data")
    data = pd.read_csv("Data//train.csv")
    update_df= pd.read_csv("Data//train_updates_20220929.csv")
    print(f"Loaded training set with {len(data)} examples")
    
    print("Preprocessing Data")
    data = Preprocessing.apply_corrections(data,update_df)
    data.pop('data_source')
    feature_names = data.columns.tolist()


# ------------------------------------------- # 
# Preprocessing
# ------------------------------------------- # 

if options['preprocessing']:
    print("\nPhase 02:\nCreating Test / Dev Sets")
    
    # Configuration options
    test_set_fraction = 0.1
    
    
    # Generate dataset
    
    train_data, test_data = Preprocessing.split_test_dev(data, test_set_fraction)
    
    train_x, train_y, train_z = Preprocessing.apply_encoding(train_data, **options)
    test_x, test_y, test_z = Preprocessing.apply_encoding(test_data, **options)
    
    del data
    
    print(f"Train set size is {train_x.shape}")
    print(f"Test set size is {test_x.shape}")
    
    # Convert to tensor
    test_x  = tf.one_hot(test_x, 27, axis = 1)
    # test_x  = tf.convert_to_tensor(test_x, dtype=tf.float32)
    test_y  = tf.convert_to_tensor(test_y, dtype=tf.float32)
    # test_z  = tf.convert_to_tensor(test_z, dtype=tf.float32)
    
    train_x = tf.one_hot(train_x, 27, axis = 1)
    # train_x = tf.convert_to_tensor(train_x, dtype=tf.float32)
    train_y = tf.convert_to_tensor(train_y, dtype=tf.float32)
    # train_z = tf.convert_to_tensor(train_z, dtype=tf.float32)
    
    # Create tensorflow dataset:
    test_set  =  tf.data.Dataset.from_tensor_slices((test_x, test_y ))
    train_set =  tf.data.Dataset.from_tensor_slices((train_x,train_y))


# ------------------------------------------- # 
# Training
# ------------------------------------------- # 

if options['train_model']:
    
    print("\nPhase 03:\nModel Training")
    
    model = tmf(train_x, train_y, test_x, test_y)
    
    # Initialize Variables 
    # parameters = Model.initialize_parameters()
    
    # Run Model
    # parameters, costs, train_acc, test_acc = Model.model_trainer(train_set,test_set, num_epochs=100)

