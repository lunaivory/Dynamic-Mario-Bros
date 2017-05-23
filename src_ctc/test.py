
import tensorflow as tf
import numpy as np
import models
import matplotlib.pyplot as plt
import time as time
import os

'''####################################################'''
'''#                Self-defined library              #'''
'''####################################################'''
from constants import *
from util import input_pipeline



graph = tf.Graph()

with graph.as_default():

	''' place holders'''
	train_samples_op, train_labels_op = input_pipline(TRAIN_FILENAMES, 'Train')
	validation_samples_op, validation_labels_op = input_pipline(VALIDATION_FILENAMES, 'Validation')

	mode = tf.placeholder(tf.bool, name='mode') # Pass True in when it is in the trainging mode
	input_samples_op, input_labels_op, input_seq_op = tf.cond(
												mode, 
												lambda: (train_samples_op, train_labels_op, np.asarray([CLIPS_PER_VIDEO] * BATCH_SIZE)), 
												lambda: (validation_samples_op, validation_labels_op, np.asarray([CLIPS_PER_VIDEO] * BATCH_SIZE))
	)

	
	# sparse tensor required by ctc_loss
	# target_samples_op = tf.sparse_placeholder(tf.int32)
	target_sample_op = sparse_tuple_from(input_labels_op)

	# seq_len = tf.placeholder(tf.int32, [BATCH_SIZE])
	seq_len = input_seq_op


	'''Define the cells'''
	logits = models.conv_model_with_layers_api(input_samples_op, FLAGS.dropout_rate, mode)

	loss = tf.nn.ctc_loss(input_labels_op, logits, input_seq_op, time_major=False)
	cost = tf.reduce_mean(loss)

	learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
																						decay_steps = 1 * FLAGS.epoch_length, decay_rate=0.96, stiarecase=True)
	optimizer = tf.train.AdamOptimizer(learning_rate)

with tf.Session(graph=graph) as sess:
	# set up the queue
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	sess.run(init_op)

	for epoch in range(1, NUM_EPOCHS + 1):
		train_cost = 0
		start = time.time()
		
		for batch in range(len(TRAIN_FILENAMES) / BATCH_SIZE):
				
			try:
					if coord.should_stop():
							break

				#feed_dict = {input_sample_op : train_inputs,
				#						 target_sample_op: train_labels,
				#						 seq_len: np.asarray([CLIPS_PER_VIDEO]*BATCH_SIZE))
				feed_dict = {mode: True}

				batch_cost, _ = sess.run([cost, optimizer], feed_dict)
				train_cost += batch_cost * batch_size

			except Exception, e:
					coord.request_stop(e)
			finally:
					coord.request_stop()
					coord.join(threads)


	train_cost /= len(TRAIN_FILENAMES)
	print('[%d/%d] [Training] Loss : %.3f' % (epoch, epoch * len(TRAIN_FILENAMES)/BATCH_SIZE + batch, train_loss))

