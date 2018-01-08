
import tensorflow as tf
import numpy as np 
import os

def logits_layer(X, n_classes):
    ''' This is just a straight forward linear layer - returns logits'''
    n_input = X.shape[1]  # (?, features)
    with tf.variable_scope('logits'):
        W =  tf.get_variable("W", shape = [n_input, n_classes],
                             initializer= tf.contrib.layers.xavier_initializer())
        b =  tf.get_variable("b", shape = [1, n_classes],
                             initializer = tf.zeros_initializer()) 
        logits = tf.add(tf.matmul(X,W), b)

    return logits

def relu(X, n_neurons, name_scope):
    ''' 
    Insert docstring
    '''
    assert type(name_scope) == str
    n_input = X.shape[1]  # (?, features)
    with tf.variable_scope(name_scope):
        W  =  tf.get_variable("W", shape =[n_input, n_neurons],
                             initializer=tf.contrib.layers.xavier_initializer())
        b  =  tf.get_variable("b", shape =[1, n_neurons],
                              initializer = tf.zeros_initializer()) 
        
        preactivations = tf.add(tf.matmul(X,W), b)  # X dims are batch x n_neurons
        activations    = tf.nn.relu(preactivations)

    for v in [W, b]:
        tf.add_to_collection("summary_weights", v)
    for v in [preactivations, activations]:
        tf.add_to_collection("activations", v) 

    return activations

def batch_relu(X, n_neurons, training_bool, name_scope):
    ''' 
    Insert docstring

    ** Todo: bias here is pointless no?
    '''
    assert type(name_scope) == str
    n_input = X.shape[1]  # (?, features)
    with tf.variable_scope(name_scope):
        W  =  tf.get_variable("W", shape =[n_input, n_neurons],
                             initializer=tf.contrib.layers.xavier_initializer())
        b  =  tf.get_variable("b", shape =[1, n_neurons],
                              initializer = tf.zeros_initializer()) 
        
        preactivations = tf.add(tf.matmul(X,W), b)    # X dims are batch x n_neurons
        bn_preactivations = tf.contrib.layers.batch_norm(preactivations,
                                                         center = True,
                                                         scale = True,
                                                         is_training=training_bool,
                                                         scope='bn')
        activations    = tf.nn.relu(bn_preactivations)

    for v in [W, b]:
        tf.add_to_collection("summary_weights", v)
    for v in [preactivations, activations, bn_preactivations]:
        tf.add_to_collection("activations", v) 

    return activations

def layer_norm_relu(X, n_neurons, name_scope):
    ''' 
    Insert docstring

    ** Todo: bias here is pointless no?
    '''
    assert type(name_scope) == str
    n_input = X.shape[1]  # (?, features)
    with tf.variable_scope(name_scope):
        W  =  tf.get_variable("W", shape =[n_input, n_neurons],
                             initializer=tf.contrib.layers.xavier_initializer())
        b  =  tf.get_variable("b", shape =[1, n_neurons],
                              initializer = tf.zeros_initializer()) 
        
        preactivations = tf.add(tf.matmul(X,W), b)    # X dims are batch x n_neurons
        ln_preactivations = tf.contrib.layers.batch_norm(preactivations,
                                                         center = True,
                                                         scale = True,
                                                         scope='ln')
        activations    = tf.nn.relu(ln_preactivations)

    for v in [W, b]:
        tf.add_to_collection("summary_weights", v)
    for v in [preactivations, activations, ln_preactivations]:
        tf.add_to_collection("activations", v) 

    return activations

def shunted_relu(X, n_neurons, n_inhib, name_scope):
    assert type(name_scope) == str
    n_input = X.shape[1].value  # (?, features)
    with tf.variable_scope(name_scope):
        We  =  tf.get_variable("We", shape =[n_input, n_neurons],
                               initializer =tf.contrib.layers.xavier_initializer())
        Wi  =  tf.get_variable("Wi",  initializer = tf.ones( shape =[n_input, n_inhib])   /n_input)
        Wei =  tf.get_variable("Wei", initializer = tf.ones( shape = [n_inhib, n_neurons])/n_inhib)
        be  =  tf.get_variable("be", shape =[1,n_neurons],
                             initializer = tf.zeros_initializer())
        bi  = tf.get_variable("bi", shape =[1,n_inhib],
                             initializer = tf.zeros_initializer())
        
        wtX = tf.matmul(X,We, name = "excitatory_pre_shunt_activations")
        zi  = tf.add(tf.matmul(X, Wi), bi, name="shunt_preactivations")
        shunt_activations = tf.nn.relu(zi, name='shunt_activations')
        shunt = tf.add(tf.matmul(shunt_activations,Wei), 1e-4, name='shunt_final')
        preactivations = tf.add(tf.divide(wtX, shunt), be,     name='preactivations')
        #preactivations = tf.contrib.layers.layer_norm(preactivations, center = True, scale = True, scope='layer_norm')
        activations = tf.nn.relu(preactivations, name='activations')
        tf.add_to_collection("constrained_weights", Wei)
        #tf.add_to_collection("constrained_weights", Wi)
        #tf.add_to_collection("constrained_weights", We)
        
        for v in [Wei, Wi, We]:
            tf.add_to_collection("summary_weights", v)
        for v in [preactivations, activations, zi, shunt_activations, shunt, wtX]:
            tf.add_to_collection("activations", v)
    return activations
