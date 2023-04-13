#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 22:09:21 2022

@author: McGillDocs
"""
import numpy as np 
import tensorflow as tf

loss =  tf.keras.losses.MeanSquaredError()

# initialize weights and biases
def initialize_parameters() :
    
    # Initializer
    initializer = tf.keras.initializers.GlorotNormal()
    
    # Weights
    W1 = tf.Variable(initializer(shape = (300,9000)), dtype=tf.float32) 
    b1 = tf.Variable(initializer(shape = (300,1)), dtype=tf.float32) 
    W2 = tf.Variable(initializer(shape = (50,300)), dtype=tf.float32) 
    b2 = tf.Variable(initializer(shape = (50,1)), dtype=tf.float32) 
    W3 = tf.Variable(initializer(shape = (20,50)), dtype=tf.float32) 
    b3 = tf.Variable(initializer(shape = (20,1)), dtype=tf.float32) 
    W4 = tf.Variable(initializer(shape = (1,20)), dtype=tf.float32) 
    b4 = tf.Variable(initializer(shape = (1,1)), dtype=tf.float32) 
    
    
    parameters = { "W1" : W1,
                   "b1" : b1,
                   "W2" : W2,
                   "b2" : b2,
                   "W3" : W3,
                   "b3" : b3,
                   "W4" : W4,
                   "b4" : b4,
                 }
    
    return parameters

# Implement forward pass

def forward_propagation(X,parameters):
    
    
    Z1 = tf.add(tf.linalg.matmul(parameters['W1'], X), parameters['b1'])
    A1 = tf.keras.activations.relu(Z1)
    
    Z2 = tf.add(tf.linalg.matmul(parameters['W2'], A1), parameters['b2'])
    A2 = tf.keras.activations.relu(Z2)
    
    Z3 = tf.add(tf.linalg.matmul(parameters['W3'], A2), parameters['b3'])
    A3 = tf.keras.activations.relu(Z3)
    
    Z4 = tf.add(tf.linalg.matmul(parameters['W4'], A3), parameters['b4'])
    A4 = tf.keras.activations.relu(Z4)
    
    return A4

# Loss function
def cost_function(y_pred, y_true):
    
    return loss(y_pred, y_true)

def model_trainer(train_dataset, test_dataset, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 1000, print_cost = True):
    
    costs = []
    train_acc = []
    test_acc = []
    
    # initialize params
    parameters = initialize_parameters()
    
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    # Accuracy
    test_accuracy = tf.keras.metrics.RootMeanSquaredError()
    train_accuracy = tf.keras.metrics.RootMeanSquaredError()
    
    # zip dataset
    #train_dataset = tf.data.Dataset.zip((X_train,Y_train))
    #test_dataset = tf.data.Dataset.zip((X_test,Y_test))
    
    # Get dataset size 
    m = train_dataset.cardinality().numpy()
    
    #Make mini batches if required
    
    train_minibatches = train_dataset.batch(minibatch_size).prefetch(8)
    test_minibatches = test_dataset.batch(minibatch_size).prefetch(8)
    
    for epoch in range(num_epochs):
        
        epoch_cost = 0
        
        train_accuracy.reset_states()
        
        for (minibatch_x,minibatch_y) in train_minibatches:
            
            with tf.GradientTape() as tape:
                y_pred = forward_propagation(tf.transpose(minibatch_x) , parameters)
                
                minibatch_cost = cost_function(tf.transpose(y_pred),(minibatch_y))
            
            # update accuracy
            train_accuracy.update_state(minibatch_y, tf.greater(tf.transpose(y_pred),0.5))
            
            # Compute weights
            trainable_variables = [parameters['W1'],
                                   parameters['b1'],
                                   parameters['W2'],
                                   parameters['b2'],
                                   parameters['W3'],
                                   parameters['b3'],
                                   parameters['W4'],
                                   parameters['b4'] ]
            
            grads = tape.gradient(minibatch_cost, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            
            epoch_cost += minibatch_cost
        
        # normalize cost
        epoch_cost /= m
    
        # print results
        if (print_cost == True) and (epoch % 10 == 1):
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            print("Training accuracy:", train_accuracy.result())
            
        # compute test cost
        
            for (minibatch_x,minibatch_y) in test_minibatches:
                y_pred = forward_propagation(tf.transpose(minibatch_x) , parameters)
                test_accuracy.update_state(minibatch_y, tf.math.greater(tf.transpose(y_pred),0.5))
            
            print("Test_accuracy:", test_accuracy.result())

            #append results 
            costs.append(epoch_cost)
            train_acc.append(train_accuracy.result())
            test_acc.append(test_accuracy.result())
            test_accuracy.reset_states()
            
    return parameters, costs, train_acc, test_acc
     
    