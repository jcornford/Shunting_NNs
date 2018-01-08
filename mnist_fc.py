import os
import sys

import numpy as np
import tensorflow as tf


import tf_layers as nn
from tf_utils import get_run_index, get_clip_op, get_summary_op, get_activations_summary_op

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def run_mnist_model(model_type, normalisation = None):
	''' model_type = 'relu' or 'shunted_relu' '''  
	# MNIST model Parameters
	n_input = 784         # MNIST data input (img shape: 28*28)
	n_classes = 10        # MNIST total classes (0-9 digits)
	learning_rate = 1e-4
	batch_size = 2**7 # 128
	training_epochs = 7
	n_batch_per_epoch = int(np.ceil(mnist.train.num_examples/batch_size))
	root_dir = "output/"
	n_run = get_run_index(root_dir)
	tb_log_string = root_dir+n_run+'_run_'+model_type+'_norm-'+str(normalisation)

	# tf Graph input
	#phase = tf.placeholder(tf.bool, name='phase')
	#with tf.name_scope('input'):
	X = tf.placeholder("float32", [None, n_input],   name= "X")
	y = tf.placeholder("float32", [None, n_classes], name= "y")

		#x_image = tf.reshape(X, [-1, 28, 28, 1])
		#tf.summary.image('input', x_image, 3)
	training_bool = tf.placeholder(tf.bool,    name='training_bool')

	# Build model
	n_neurons = 100
	n_inhib   = 2
	if model_type == 'shunted_relu':
		h1 = nn.shunted_relu(X,  n_neurons, n_inhib, 'h1')
		h2 = nn.shunted_relu(h1, n_neurons, n_inhib, 'h2')
		hf = nn.shunted_relu(h2, n_neurons, n_inhib, 'h_final')
		tb_log_string += '_'+str(n_neurons)+'_neurons'+'_inhib_'+str(n_inhib)

		#tb_log_string +=' episoln 1'
		clip_op = get_clip_op()


		if normalisation is not None:
			print('*****************************************')
			print('No normalisation coded for shunting relu!')
			print('*****************************************')

	elif model_type == 'relu':
		clip_op = None
		tb_log_string+='_'+str(n_neurons)+'_neurons'
		if normalisation is None:
			h1 = nn.relu(X,  n_neurons, 'h1')
			h2 = nn.relu(h1, n_neurons, 'h2')
			hf = nn.relu(h2, n_neurons, 'h_final')

		elif normalisation == 'ln':
			h1 = nn.layer_norm_relu(X,  n_neurons, 'h1')
			h2 = nn.layer_norm_relu(h1, n_neurons, 'h2')
			hf = nn.layer_norm_relu(h2, n_neurons, 'h_final')

		elif normalisation == 'bn':
			print('batch_norm engaged')
			h1 = nn.batch_relu(X,  n_neurons, training_bool, 'h1')
			h2 = nn.batch_relu(h1, n_neurons, training_bool, 'h2')
			hf = nn.batch_relu(h2, n_neurons, training_bool, 'h_final')

		else:
			print('Unregcognised relu norm type!')
			return 0

	else:
		print('Unregcognised model type!')
		return 0

	logits = nn.logits_layer(hf, n_classes)

	with tf.name_scope("cross_ent"):
	    xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
	            logits=logits, labels=y), name="cross_ent")
	    tf.summary.scalar("cross_ent", xent)

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops): # add update batch norm params to training step
		#with tf.name_scope("train"):
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

	# make this and op?
	with tf.name_scope("accuracy"): # this an op?
		correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		train_summary = tf.summary.scalar("training_accuracy", accuracy)
		test_summary  = tf.summary.scalar("testing_accuracy", accuracy)

	summaries_op = get_summary_op() # rename to something that shows not depend on X
	activations_summaries_op = get_activations_summary_op()

	# run mnist code
	# Todo split into return the graph and running
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter(tb_log_string) # can update this later for hyper params
	writer.add_graph(sess.graph)

	# log the starting parameters
	summary_buffer = sess.run(summaries_op) 
	writer.add_summary(summary_buffer, global_step=0)

	print(n_batch_per_epoch, ' batches per epoch - though check not how to handle last batch?')
	for epoch in range(training_epochs):
		for i in range(n_batch_per_epoch):
			step = epoch * n_batch_per_epoch + i # steps are batch presentations

			batch = mnist.train.next_batch(batch_size)
			sess.run(train_step, feed_dict={'X:0': batch[0], 'y:0': batch[1],'training_bool:0':1})
			if clip_op is not None:
				sess.run(clip_op)

			# Todo - your handling of the accuracy ops is a little clunky / unsure what it going on exactly
			if step % 50 == 0:
				train_accuracy, train_sum = sess.run([accuracy, train_summary], feed_dict={'X:0': mnist.train.images,
																 'y:0': mnist.train.labels,'training_bool:0': 1})
				#feed_dict={'X:0': batch[0], 'y:0': batch[1], 'training_bool:0'=1})
				writer.add_summary(train_sum, step)

			if step%50 == 0:
				#test_accuracy, test_sum = sess.run([accuracy, test_summary],
				test_accuracy, test_sum = sess.run([accuracy, test_summary],
				feed_dict={'X:0': mnist.test.images, 'y:0': mnist.test.labels,'training_bool:0': 0})
				writer.add_summary(test_sum, step)

			if step%n_batch_per_epoch ==0:
				print(test_accuracy)

			if step%100 == 0:
				output = sess.run(summaries_op)
				writer.add_summary(output, step)
				activations_output = sess.run(activations_summaries_op,feed_dict={'X:0': batch[0], 'y:0': batch[1],
											 'training_bool:0':1})
				writer.add_summary(activations_output, step)

if __name__ == '__main__':
	print('running', sys.argv[0])
	arguments = sys.argv
	print(arguments)
	if len(arguments) == 2:
		run_mnist_model(model_type=sys.argv[1])
	elif len(arguments) == 3:
		run_mnist_model(model_type=sys.argv[1], normalisation=sys.argv[2])
	else:
		run_mnist_model(model_type='relu')

