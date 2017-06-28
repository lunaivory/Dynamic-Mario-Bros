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
    data_3dcnn = util_training_3dcnn.input_pipeline(constants_3dcnn.TRAIN_FILENAMES)

    #input_samples_op, input_labels_op, input_dense_label_op, input_clip_label_op = util_training.input_pipeline(TRAIN_FILENAMES)
    # Pass True in when it is in the trainging mode
    mode = tf.placeholder(tf.bool, name='mode')
    net_type = tf.placeholder(tf.bool, name='net_type')
    
    input_samples_op, input_labels_op, input_dense_label_op, input_clip_label_op = data_3dcnn

    loss_avg = tf.placeholder(tf.float32, name='loss_avg')
    accuracy_avg = tf.placeholder(tf.float32, name='accuracy_avg')

    # Returns 'logits' layer, the top-most layer of the network
    dropout2_flat = model.dynamic_mario_bros(input_samples_op, FLAGS.dropout_rate, mode)

    logits = tf.layers.dense(inputs=tf.reshape(dropout2_flat, shape=[CLIPS_PER_VIDEO,-1]), units=21)

    '''Set up Variables'''
    # Count number of samples fed and correct predictions made.
    # Attached to a summary op
    global_step = tf.Variable(1, name='global_step', trainable=False)

    counter_correct_prediction = tf.Variable(0, name='counter_correct_prediction', trainable=False)
    counter_samples_fed = tf.Variable(0, name='counter_samples_fed', trainable=False)

    # Loss calculations: cross-entropy
    with tf.name_scope('ctc_loss'):
        # Return : A 1-D float tensor of shape [1]
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_clip_label_op, logits=tf.squeeze(logits)))

    # Accuracy calculations
    with tf.name_scope('accuracy'):
        
        logits = tf.reshape(logits, shape=[-1,21])
        predictions = tf.argmax(logits, 1, name='predictions')
        predictions = tf.Print(predictions,[predictions], summarize=30, message='CNN')
        input_clip_label_op = tf.Print(input_clip_label_op, [input_clip_label_op], summarize=30, message='REF')

        logits_softmax = tf.nn.softmax(logits)
        correct_predictions = tf.nn.in_top_k(logits_softmax, input_clip_label_op, 1)
        batch_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        num_correct_predictions = tf.reduce_sum(tf.cast(correct_predictions, tf.int32))

    # Create optimization op.
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, name='Adam_3dcnn')
        train_op = optimizer.minimize(loss, global_step=global_step)


    tf.add_to_collection('predictions', predictions)
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
        try:
            for epoch in range(1, FLAGS.num_epochs + 1):

                for i in range(FLAGS.epoch_length):
                    if coord.should_stop():
                        break

                    step = tf.train.global_step(sess, global_step)
                  
                    if (step % FLAGS.checkpoint_every_step) == 0:
                        ckpt_save_path = saver.save(sess, os.path.join(FLAGS.model_dir, 'model'), global_step)
                        print('CNN Model saved in file : %s' % ckpt_save_path)

                    feed_dict = {mode: True, net_type: net_ty}
                    request_output = [num_correct_predictions, predictions, input_clip_label_op, loss, train_op]
                    correct_predictions_training, preds, true_labels, loss_training, _ = sess.run(
                    request_output, feed_dict=feed_dict)
                    
                    ### Update counters
                    counter_correct_predictions_training += correct_predictions_training
                    # counter_correct_predictions_training += jaccard_similarity_score(true_labels, np.argmax(predictions, axis=1))
                    counter_loss_training += loss_training

                    # Print status message.
                    if (step % FLAGS.print_every_step == 0):
                        accuracy_avg_value_training = counter_correct_predictions_training / (FLAGS.print_every_step*BATCH_SIZE*CLIPS_PER_VIDEO)
                        loss_avg_value_training = counter_loss_training / (FLAGS.print_every_step)
                        prev_accuracy = accuracy_avg_value_training
                        print('[%d/%d] [Training] Accuracy: %.3f, Loss: %.3f\n\n' % (epoch, step, accuracy_avg_value_training, loss_avg_value_training), flush=True)

                        # Reset counters
                        counter_correct_predictions_training = 0.0
                        counter_loss_training = 0.0


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
