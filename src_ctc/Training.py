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


'''#########################################################'''
'''#                Create Network                         #'''
'''#########################################################'''

''' Set Up Flags'''
# Model Hyperparameters
tf.flags.DEFINE_float('dropout_rate', DROPOUT_RATE, 'Dropout rate (default: 0.5)')
# Training Parameters
tf.flags.DEFINE_integer('learning_rate', LEARNING_RATE, 'Batch Size (default: 1e-3)')
tf.flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Batch Size (default: 32)')
tf.flags.DEFINE_integer('num_epochs', NUM_EPOCHS, 'Number of full passess over whole training data (default: 100)')
tf.flags.DEFINE_integer('print_every_step', PRINT_EVERY_STEP, 'Print training details after this many steps/iterations (i.e., batches) (default: 10)')
tf.flags.DEFINE_integer('evaluate_every_step', EVALUATE_EVERY_STEP, 'Evaluate model on validation set after this many steps/iterations (i.e., batches) (default: 500)')
tf.flags.DEFINE_integer('checkpoint_every_step', CHECKPOINT_EVERY_STEP, 'Save model after this many steps/iterations (i.e., batches) (default: 1000)')
tf.flags.DEFINE_string('log_dir', './runs/', 'Output directory (default: "./runs/")')
tf.flags.DEFINE_integer('epoch_length', int(len(TRAIN_FILENAMES)/tf.flags.FLAGS.batch_size), 'Batch Size (default: 1e-3)')
tf.flags.DEFINE_integer('validation_length', int(len(VALIDATION_FILENAMES)/tf.flags.FLAGS.batch_size), 'validation iteration num')

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


''' set up placeholders '''
# Feed a batch of training data at each training step using the {feed_dict} argument in sess.run()
train_samples_op, train_labels_op = input_pipline(TRAIN_FILENAMES, 'Train')
validation_samples_op, validation_labels_op = input_pipline(VALIDATION_FILENAMES, 'Validation')

mode = tf.placeholder(tf.bool, name='mode') # Pass True in when it is in the trainging mode
loss_avg = tf.placeholder(tf.float32, name='loss_avg')
accuracy_avg = tf.placeholder(tf.float32, name='accuracy_avg')
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

# Try this first, if not working check this : http://stackoverflow.com/questions/41162955/tensorflow-queues-switching-between-train-and-validation-data
# TODO : Check if this is working!!!
input_samples_op, input_labels_op, input_seq_op = tf.cond(mode, lambda: (train_samples_op, train_labels_op, [CLIPS_PER_VIDEO] * BATCH_SIZE), 
                                                                lambda: (validation_samples_op, validation_labels_op, [CLIPS_PER_VIDEO] * BATCH_SIZE))
# pass in parameters that controls external inputs
# Returns 'logits' layer, the top-most layer of the network
logits = models.conv_model_with_layers_api(input_samples_op, FLAGS.dropout_rate, mode)


'''Set up Variables'''
# Count number of samples fed and correct predictions made.
# Attached to a summary op
global_step = tf.Variable(1, name='global_step', trainable=False)

counter_correct_prediction = tf.Varialbe(0, name='counter_correct_prediction', trainable=False)
counter_samples_fed = tf.Variable(0, name='counter_samples_fed', trainable=False)

# Loss calculations: cross-entropy
with tf.name_scope('cross_entropy_loss'):
    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=input_labels_op))
    # inputs : 3-D float Tensor
    # logits : an int32 sparseTensor
    # Time_major : If false then inputs shape is [batch_size, time_size, ...], else [time_size, batch_size, ...]
    # sequence_len : 1D int32 shape [batch_size] tensor, the sequence length (Constant in our case?)
    # Return : A 1-D float tensor of shape [batch]
    # TODO : Check ctc_greedy_decoder
    loss = tf.nn.ctc_loss(input_labels_op, logits, input_seq_op, time_major=False)

# Accuracy calculations
with tf.name_scope('accuracy'):
    predictions = tf.argmax(logits, 1, name='predictions')
    predictions = tf.Print(predictions,[predictions])
    correct_predictions = tf.nn.in_top_k(logits, input_labels_op, 1)
    batch_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    num_correct_predictions = tf.reduce_sum(tf.cast(correct_predictions, tf.int32))

# Create optimization op.
with tf.name_scope('train'):
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                              decay_steps = 1 * FLAGS.epoch_length, decay_rate=0.96, stiarecase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step = global_step)

tf.add_to_collection('predictions', predictions)
tf.add_to_collection('input_samples_op', input_samples_op)
tf.add_to_collection('mode', mode)

def do_evaluation(sess, samples, labels):
    counter_accuracy = 0.0
    counter_loss = 0.0
    counter_batches = 0
    for i in range(FLAGS.validation_length):
        counter_batches += 1
        feed_dict = {mode:False}
        results = sess.run([loss, num_correct_predictions], feed_dict=feed_dict)
        counter_loss += result[0]
        counter_accuracy += result[1]
    return (counter_loss/counter_batches, counter_accuracy/(counter_batches * FLAGS.batch_size))


''' Create Session '''
sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)


''' Summary record '''
# Crate summary op for monitoring the training. Each summary op annotates a node in the computational graph 
# and collects data from it.
summary_train_loss = tf.summary.scalar('loss', loss)
summary_train_acc = tf.summary.scalar('accuracy_training', batch_accuracy)
summary_avg_accuracy = tf.summary.scalar('accuracy_avg', accuracy_avg)
summary_avg_loss = tf.summary.scalar('loss_avg', loss_avg)
summary_learning_rate = tf.summary.scalar('learning_rate', learning_rate)
summary_images = tf.summary.image('images', input_samples_op, max_outputs=10)

# Group summaries.
summaries_training = tf.summary.merge([summary_trian_loss, summary_train_acc, summary_learning_rate, summary_images])#, summary_model])
summaries_evaluation = tf.summary.merge([summary_avg_accuracy, summary_avg_loss])

# Register summary ops.
train_summary_dir = os.path.join(FLAGS.model_dir, "summary", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

valid_summary_dir = os.path.join(FLAGS.model_dir, "summary", "validation")
valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)

# Create a saver for writing training checkpoints.
saver = tf.train.Saver(max_to_keep=3)


'''###############################################'''
'''#       Create Training  Routine              #'''
'''###############################################'''

# Define counters for accumulating measurements
counter_correct_predictions_training = 0.0
counter_loss_training = 0.0

# set up the queue
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for epoch in range(1, FLAGS.num_epochs + 1):
    for i in range(FLAGS.epoch_length):
        
        step = tf.train.global_step(sess, global_step)
        
        if (step % FLAGS.checkpoint_every_step) == 0:
            ckpt_save_path = saver.save(sess, os.path.join(FLAGS.model_dir, 'model'), global_step)
            print('Model saved in file : %s' % ckpt_save_path)
            
        try:
            if coord.should_stop():
                break

            # Training
            feed_dict = {mode: True}
            ### only the operations that are fed are evaluated
            request_output = [summaries_training, num_correct_predictions, loss, train_op]
            train_summary, correct_predictions_trainiing, loss_training, _ = sess.run(request_output, feed_dict=feed_dict)
            # TODO: count correct prediction are Jacob error function????
            ### Update counters
            counter_correct_predictions_training += correct_predictions_training
            counter_loss_training += loss_training
            ### Write summary data
            train_summary_writer.add_summary(train_summary, step)

            # Print status message.
            if (step % FLAGS.print_every_step) == 0:
                accuracy_avg_value_training = counter_correct_predictions_training / (FLAGS.print_every_step * FLAGS.batch_size)
                loss_avg_value_training = counter_loss_training/FLAGS.print_every_step
                print('[%d/%d] [Training] Accuracy: %.3f, Loss: %.3f' % (epoch, step, accuracy_avg_value_training, loss_avg_value_training))
                # Reset counters
                counter_correct_predictions_training = 0.0
                counter_loss_training = 0.0
                # Report : Note that accuracy_avg and loss_avg placeholders are defined just to feed average results to summaries.
                summary_report = sess.run(summaries_evaluation, feed_dict={accuracy_avg:accuracy_avg_value_training, loss_avg:loss_avg_value_training})
                train_summary_writer.add_summary(summary_report, step)
            
            # Validation
            if (step%FLAGS.evaluate_every_step) == 0:
                (loss_avg_value_validation, accuracy_avg_value_validation) = do_evaluation(sess, validation_data, validation_labels)
                print('[%d/%d] [Validation] Accuracy: %.3f, Loss: %.3f' % (epoch, step, accuracy_avg_value_validation, loss_avg_value_validation))
                # Report
                summary_report = sess.run(summaries_evaluation, feed_dict={accuracy_avg:accuracy_avg_value_validation, loss_avg:loss_avg_value_validation})
                valid_summary_writer.add_summary(summary_report, step)
            
        except Exception, e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':

    tf.app.run()
