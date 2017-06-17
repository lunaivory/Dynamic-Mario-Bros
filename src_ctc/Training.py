import tensorflow as tf
import numpy as np
import model
import matplotlib.pyplot as plt
import time as time
import os

import util_training
import util_training_3dcnn

'''####################################################'''
'''#                Self-defined library              #'''
'''####################################################'''
from constants import *
import constants_3dcnn
from sklearn.metrics import jaccard_similarity_score

'''#########################################################'''
'''#                Create Network                         #'''
'''#########################################################'''

''' Set Up Flags'''
# Model Hyperparameters
tf.flags.DEFINE_float('dropout_rate', DROPOUT_RATE, 'Dropout rate (default: 0.5)')

# Training Parameters
tf.flags.DEFINE_integer('learning_rate', LEARNING_RATE, 'Batch Size (default: 1e-3)')
tf.flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Batch Size (default: 5)')
tf.flags.DEFINE_integer('num_epochs', NUM_EPOCHS, 'Number of full passess over whole training data (default: 100)')
tf.flags.DEFINE_integer('epoch_length', int(len(TRAIN_FILENAMES)/tf.flags.FLAGS.batch_size), 'Batch Size (default: 1e-3)')
tf.flags.DEFINE_integer('validation_length', int(len(VALIDATION_FILENAMES)/tf.flags.FLAGS.batch_size), 'validation iteration num')
tf.flags.DEFINE_integer('evaluate_every_step', EVALUATE_EVERY_STEP, 'Evaluate model on validation set after this many steps/iterations (i.e., batches) (default: 500)')

# Log parameters
tf.flags.DEFINE_integer('print_every_step', PRINT_EVERY_STEP, 'Print training details after this many steps/iterations (i.e., batches) (default: 10)')
tf.flags.DEFINE_integer('checkpoint_every_step', CHECKPOINT_EVERY_STEP, 'Save model after this many steps/iterations (i.e., batches) (default: 1000)')
tf.flags.DEFINE_string('log_dir', './runs/', 'Output directory (default: "./runs/")')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print('\nCommand-line Arguments:')
for attr, value in sorted(FLAGS.__flags.items()):
    print('{}={}'.format(attr.upper(), value))
print('')

''' Create Directory for this experiment'''
timestamp = str(int(time.time()))
FLAGS.model_dir = os.path.abspath(os.path.join(FLAGS.log_dir, timestamp))
print("Writing to {}\n".format(FLAGS.model_dir))

graph = tf.Graph()

with graph.as_default():
    ''' set up placeholders '''
    # Training and validation placeholders
    data_lstm = util_training.input_pipeline(TRAIN_FILENAMES)
    data_3dcnn = util_training_3dcnn.input_pipeline(constants_3dcnn.TRAIN_FILENAMES)

    #input_samples_op, input_labels_op, input_dense_label_op, input_clip_label_op = util_training.input_pipeline(TRAIN_FILENAMES)
    # Pass True in when it is in the trainging mode
    mode = tf.placeholder(tf.bool, name='mode')
    net_type = tf.placeholder(tf.bool, name='net_type')
    
    input_samples_op, input_labels_op, input_dense_label_op, input_clip_label_op = tf.cond(
                                     net_type,
                                     lambda : data_lstm,
                                     lambda : data_3dcnn)

    loss_avg = tf.placeholder(tf.float32, name='loss_avg')
    accuracy_avg = tf.placeholder(tf.float32, name='accuracy_avg')
    # learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    # Returns 'logits' layer, the top-most layer of the network
    dropout2_flat = model.dynamic_mario_bros(input_samples_op, FLAGS.dropout_rate, mode)

    #logits = tf.layers.dense(inputs=dropout2_flat, units=21)

    cnn_representations = tf.map_fn(lambda x: model.dynamic_mario_bros(input_samples_op, FLAGS.dropout_rate, mode, reuse=True),
                                         elems=dropout2_flat,
                                         dtype=tf.float32,
                                         back_prop=True)

    logits = tf.layers.dense(inputs=tf.reshape(cnn_representations, shape=[CLIPS_PER_VIDEO,-1]), units=21)

    cnn_representations = tf.reshape(cnn_representations, shape=[1, CLIPS_PER_VIDEO, -1])
    # lstm
    with tf.name_scope("LSTM"):
        seq_length =CLIPS_PER_VIDEO # tf.shape(dropout2_flat)[1]
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=LSTM_HIDDEN_UNITS)
        lstm_outputs, _ = tf.nn.dynamic_rnn(lstm_cell, dropout2_flat, dtype=tf.float32, time_major=False, sequence_length=[seq_length])

    # Add dropout operation
    with tf.name_scope("dropout3"):
        dropout3 = tf.layers.dropout(inputs=lstm_outputs, rate=DROPOUT_RATE, training=mode)

    # Dense Layer
    with tf.name_scope("dense3"):
        dense3 = tf.layers.dense(inputs=dropout3, units=LSTM_HIDDEN_UNITS, activation=tf.nn.relu,
                                 #kernel_regularizer=slim.l2_regularizer(weight_decay),
                                 #bias_regularizer=slim.l2_regularizer(weight_decay)
                                 )

    # Add dropout operation
    with tf.name_scope("dropout4"):
        dropout4 = tf.layers.dropout(inputs=dense3, rate=DROPOUT_RATE, training=mode)

    with tf.name_scope("logits"):
        logits_lstm = tf.layers.dense(inputs=dropout4, units=21)

    '''Set up Variables'''
    # Count number of samples fed and correct predictions made.
    # Attached to a summary op
    global_step = tf.Variable(1, name='global_step', trainable=False)
    global_step_lstm = tf.Variable(1, name='global_step', trainable=False)

    counter_correct_prediction = tf.Variable(0, name='counter_correct_prediction', trainable=False)
    counter_samples_fed = tf.Variable(0, name='counter_samples_fed', trainable=False)

    # Loss calculations: cross-entropy
    with tf.name_scope('ctc_loss'):
        # Return : A 1-D float tensor of shape [1]
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_clip_label_op, logits=tf.squeeze(logits)))

        loss_lstm = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_clip_label_op, logits=tf.squeeze(logits_lstm)))
        #loss_ctc = tf.reduce_mean(tf.nn.ctc_loss(input_labels_op, logits_ctc, sequence_length=[CLIPS_PER_VIDEO*FLAGS.batch_size], time_major=False))

    # Accuracy calculations
    with tf.name_scope('accuracy'):
        
        logits = tf.reshape(logits, shape=[-1,21])
        predictions = tf.argmax(logits, 1, name='predictions')
        predictions = tf.Print(predictions,[predictions], summarize=30, message='CNN')
        input_clip_label_op = tf.Print(input_clip_label_op, [input_clip_label_op], summarize=30, message='REF')

        logits_lstm = tf.reshape(logits_lstm, shape=[-1, 21])
        predictions_lstm = tf.argmax(logits_lstm, 1, name='predictions')
        predictions_lstm = tf.Print(predictions_lstm, [predictions_lstm], summarize=30, message='RNN')

        logits_softmax = tf.nn.softmax(logits)
        logits_softmax_lstm = tf.nn.softmax(logits_lstm)
        #logits_expanded = tf.stack([tf.squeeze(tf.nn.softmax(logits)) for i in range(FRAMES_PER_CLIP)], axis=1)
        #logits_expanded = tf.reshape(logits_expanded, [FRAMES_PER_VIDEO*BATCH_SIZE, NO_GESTURE])
        correct_predictions = tf.nn.in_top_k(logits_softmax, input_clip_label_op, 1)
        batch_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        num_correct_predictions = tf.reduce_sum(tf.cast(correct_predictions, tf.int32))

        correct_predictions_lstm = tf.nn.in_top_k(logits_softmax_lstm, input_clip_label_op, 1)
        batch_accuracy_lstm = tf.reduce_mean(tf.cast(correct_predictions_lstm, tf.float32))
        num_correct_predictions_lstm = tf.reduce_sum(tf.cast(correct_predictions_lstm, tf.int32))


    # Create optimization op.
    with tf.name_scope('train'):
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                decay_steps = 3 * FLAGS.epoch_length, decay_rate=0.5, staircase=True)

        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, name='Adam_3dcnn')
        train_op = optimizer.minimize(loss, global_step=global_step)

        optimizer_lstm = tf.train.AdamOptimizer(FLAGS.learning_rate, name='Adam_lstm')
        #optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        #gradients, v = zip(*optimizer_lstm.compute_gradients(loss_lstm))
        
        #clipped_gradients, _ = tf.clip_by_global_norm(gradients, 10)
        #train_op_lstm = optimizer_lstm.apply_gradients(zip(clipped_gradients, v), global_step=global_step_lstm)
        train_op_lstm = optimizer_lstm.minimize(loss, global_step=global_step_lstm)

    tf.add_to_collection('predictions', predictions)
    tf.add_to_collection('predictions_lstm', predictions_lstm)
    tf.add_to_collection('net_type', net_type)
    tf.add_to_collection('input_samples_op', input_samples_op)
    tf.add_to_collection('mode', mode)

    #with tf.Session(graph=graph) as sess:
    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 1})) as sess:
        ''' Create Session '''
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        ''' Summary record '''
        # Crate summary op for monitoring the training. Each summary op annotates a node in the computational graph
        # and collects data from it.
        summary_train_loss = tf.summary.scalar('loss', loss)
        summary_train_acc = tf.summary.scalar('accuracy_training', batch_accuracy)
        summary_avg_accuracy = tf.summary.scalar('accuracy_avg', accuracy_avg)
        summary_avg_loss = tf.summary.scalar('loss_avg', loss_avg)
        summary_learning_rate = tf.summary.scalar('learning_rate', FLAGS.learning_rate)

        # Group summaries.
        summaries_training = tf.summary.merge([summary_train_loss, summary_train_acc, summary_learning_rate])
        summaries_evaluation = tf.summary.merge([summary_avg_accuracy, summary_avg_loss])

        # Register summary ops.
        train_summary_dir = os.path.join(FLAGS.model_dir, "summary", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        valid_summary_dir = os.path.join(FLAGS.model_dir, "summary", "validation")
        valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver(max_to_keep=1)


        '''###############################################'''
        '''#       Create Training  Routine              #'''
        '''#       Create Training  Routine              #'''
        '''###############################################'''

        # Define counters for accumulating measurements
        counter_correct_predictions_training = 0.0
        counter_loss_training = 0.0
        net_ty = False
        switch = 0
        try:
            for epoch in range(1, FLAGS.num_epochs + 1):

                for i in range(FLAGS.epoch_length):
                    if coord.should_stop():
                        break

                    step = tf.train.global_step(sess, global_step)
                    step_lstm = tf.train.global_step(sess, global_step_lstm)
                  
                    if (step % FLAGS.checkpoint_every_step) == 0:
                        ckpt_save_path = saver.save(sess, os.path.join(FLAGS.model_dir, 'model'), global_step)
                        print('Model saved in file : %s' % ckpt_save_path)
                    if (step_lstm % FLAGS.checkpoint_every_step) == 0:
                        ckpt_save_path = saver.save(sess, os.path.join(FLAGS.model_dir, 'modelLSTM'), global_step_lstm)

                    # Training
                    # Training
                    feed_dict = {mode: True}
                    if (epoch > 30):
                        net_ty = True
                        feed_dict = {mode: True, net_type: net_ty}
                        request_output = [summaries_training, num_correct_predictions_lstm, predictions_lstm, input_clip_label_op, loss_lstm, train_op_lstm]
                        train_summary, correct_predictions_training, preds, true_labels, loss_training, _ = sess.run(request_output, feed_dict=feed_dict)
#                        print(preds)
#                        print('loss_rnn: ' + str(loss_training))
                    else:
                        feed_dict = {mode: True, net_type: net_ty}
                        request_output = [summaries_training, num_correct_predictions, predictions, input_clip_label_op, loss, train_op]
                        train_summary, correct_predictions_training, preds, true_labels, loss_training, _ = sess.run(
                        request_output, feed_dict=feed_dict)
                    
                    switch += 1
                    ### Update counters
                    counter_correct_predictions_training += correct_predictions_training
                    # counter_correct_predictions_training += jaccard_similarity_score(true_labels, np.argmax(predictions, axis=1))
                    counter_loss_training += loss_training
                    ### Write summary data
                    train_summary_writer.add_summary(train_summary, step)

                    # Print status message.
                    if (step % FLAGS.print_every_step == 0) or (step_lstm % FLAGS.print_every_step == 0):
                        accuracy_avg_value_training = counter_correct_predictions_training / (FLAGS.print_every_step*BATCH_SIZE*CLIPS_PER_VIDEO)
                        loss_avg_value_training = counter_loss_training / (FLAGS.print_every_step)
                        print('[%d/%d] [Training] Accuracy: %.3f, Loss: %.3f\n\n' % (epoch, step, accuracy_avg_value_training, loss_avg_value_training), flush=True)
                        # Reset counters
                        counter_correct_predictions_training = 0.0
                        counter_loss_training = 0.0
                        # Report : Note that accuracy_avg and loss_avg placeholders are defined just to feed average results to summaries.
                        summary_report = sess.run(summaries_evaluation, feed_dict={accuracy_avg:accuracy_avg_value_training, loss_avg:loss_avg_value_training, net_type: net_ty})
                        train_summary_writer.add_summary(summary_report, step)

        except Exception as e:
            # Report exceptions to the coordinator.
            print(str(e))
            coord.request_stop(e)

        finally:
            # Terminate as usual. It is safe to call `coord.request_stop()` twice.
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':

    tf.app.run()
