
import tensorflow as tf
import numpy as np 
import os

def get_clip_op():
    with tf.name_scope('Constrain_Wei'):
        clip_op_list = []
        for weights in tf.get_collection("constrained_weights"):
            clip_op_list.append(tf.assign(weights, tf.clip_by_value(weights, clip_value_min = 0,
                                                     clip_value_max = np.infty)))
        clip_op = tf.group(*clip_op_list)
    return clip_op

def get_summary_op():
    op_list = []
    for weights in tf.get_collection("summary_weights"):
        op_list.append(tf.summary.histogram(weights.name.split(':')[0], weights))
        # here you could start sliginc the wirghts[1,:] for 2nd neuron etc
    summary_op = tf.summary.merge(op_list)
    return summary_op

def get_activations_summary_op():
    op_list = []
    for weights in tf.get_collection("activations"):
        op_list.append(tf.summary.histogram(weights.name.split(':')[0], weights))
    summary_op = tf.summary.merge(op_list)
    return summary_op

def get_batch_norm_vars():
    op_list = []
    for t in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        if 'bn' in t.name:
            op_list.append(tf.summary.histogram(t.name.split(':')[0], t))
    summary_op = tf.summary.merge(op_list)
    return summary_op

def get_run_index(root_dir):
    split_names = [s.split('_')[0] for s in os.listdir(root_dir) if not s.startswith('.')]
    run_numbers = []
    for name in split_names:
        try:
            run_numbers.append(int(name))
        except:
            pass
    if run_numbers != []:
        n_run = str(max(run_numbers)+1)
    else:
        n_run = str(1)
    return n_run

