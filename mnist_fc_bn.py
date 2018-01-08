import os
import sys

import numpy as np
import tensorflow as tf
import tf_utils

from tf_utils import relu, shunted_relu, softmax_layer, batch_relu
# will split into two utis files
from tf_utils import get_run_index, get_clip_op, get_summary_op, get_activations_summary_op

#https://github.com/dandelionmane/tf-dev-summit-tensorboard-tutorial/blob/master/mnist.py
#https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def run_mnist_model(model_type, normalisation = None):
	''' model_type = 'relu' or 'shunted_relu' '''  
	# MNIST model Parameters
	n_input = 784 # MNIST data input (img shape: 28*28)
	n_classes = 10 # MNIST total classes (0-9 digits)
	learning_rate = 1e-4
	batch_size = 100
	training_epochs = 7
	n_batch_per_epoch = int(mnist.train.num_examples/batch_size)
	root_dir = "output/"
	n_run = get_run_index(root_dir)
	tb_log_string = root_dir+n_run+'_run_'+model_type+'_norm-'+normalisation

	n_neurons = 100

	# tf Graph input
	phase = tf.placeholder(tf.bool, name='phase')
	with tf.name_scope('input'):
		X = tf.placeholder("float", [None, n_input],   name= "X")
		y = tf.placeholder("float", [None, n_classes], name="y")
		#x_image = tf.reshape(X, [-1, 28, 28, 1])
		#tf.summary.image('input', x_image, 3)

	# Build model
	if model_type == 'shunted_relu':
		h1 = shunted_relu(X, 1000, 5, 'h1')
		h_final = shunted_relu(h1, 1000, 5, 'h2_final')

	elif model_type

	elif model_type == 'relu':
		if normalisation is None:
			h1 = relu(X, 1000, 'h1')
			h_final  = relu(h1, 1000, 'h2_final')
		elif normalisation == 'bn':
			h1 = batch_relu(X, 1000, 'h1', phase)
			h_final  = batch_relu(h1, 1000, 'h2_final', phase)

	elif model_type == 'relu_4x':
		h1 = relu(X, n_neurons, 'h1')
		h2 = relu(h1, n_neurons, 'h2')
		h3 = relu(h2, n_neurons, 'h3')
		h_final  = relu(h3, n_neurons, 'h4_final')

	elif model_type == 'shunted_relu_4x':
		h1 = shunted_relu(X, n_neurons, 2, 'h1')
		h2 = shunted_relu(h1, n_neurons, 2,'h2')
		h3 = shunted_relu(h2, n_neurons, 2,'h3')
		h_final  = shunted_relu(h3, n_neurons,2, 'h4_final')

	else:
		print('Unregcognised model type!')
		return 0
	logits = softmax_layer(h_final, n_classes)

	with tf.name_scope("cross_ent"):
	    xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
	            logits=logits, labels=y), name="cross_ent")
	    tf.summary.scalar("cross_ent", xent)

	# this is for bn
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)# for batch norm
	with tf.control_dependencies(update_ops):
		with tf.name_scope("train"):
			train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

	# make this and op?
	with tf.name_scope("accuracy"): 
		correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		train_summary = tf.summary.scalar("training_accuracy", accuracy)
		test_summary  = tf.summary.scalar("testing_accuracy", accuracy)

	clip_op = get_clip_op()
	summaries_op = get_summary_op() # rename to something that shows not depend on X
	activations_summaries_op = get_activations_summary_op()
	bn_summaries_op = tf_utils.get_batch_norm_vars()

	# run code
	sess = tf.Session()
	writer = tf.summary.FileWriter(tb_log_string) # can update this later for hyper params
	writer.add_graph(sess.graph)

	sess.run(tf.global_variables_initializer())
	# log the starting parameters
	summary_buffer = sess.run(summaries_op) 
	writer.add_summary(summary_buffer, global_step=0)

	# delete this 

	# loop t
	for epoch in range(training_epochs):
		for i in range(n_batch_per_epoch):
			step = epoch * n_batch_per_epoch + i

			batch = mnist.train.next_batch(batch_size)

			if step % 5 == 0:
				train_accuracy, train_sum = sess.run([accuracy, train_summary],
				 feed_dict={'X': batch[0], 'y': batch[1],'phase':1})
				writer.add_summary(train_sum, step)

			if step%50 == 0:
				test_accuracy, test_sum = sess.run([accuracy, test_summary],
				feed_dict={'X': mnist.test.images, 'y': mnist.test.labels, 'phase': 0})
				writer.add_summary(test_sum, step)

			if step%800 ==0:
				print(test_accuracy)

			if step%150 == 0:
				# weights
				output = sess.run(summaries_op)
				writer.add_summary(output, step)
				# bn
				output = sess.run(bn_summaries_op)
				writer.add_summary(output, step)

				activations_output = sess.run(activations_summaries_op,
				 feed_dict={'X': batch[0], 'y': batch[1],'phase':1})
				writer.add_summary(activations_output, step)

			sess.run(train_step, feed_dict={'X': batch[0], 'y': batch[1],'phase':1})
			sess.run(clip_op)

	try:
		w1 = graph.get_tensor_by_name("h2/Wei:0")
	except:
		pass

if __name__ == '__main__':
	print('running', sys.argv[0])
	arguments = sys.argv
	print(arguments)
	if len(arguments) > 1:
		# insert assert model type checking
		run_mnist_model(sys.argv[1])
	else:
		print('specify model')



