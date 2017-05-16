import os
import time
import datetime
import math
import pickle
import utils
import models
import numpy as np
from tensorflow.contrib import learn
import tensorflow as tf
import matplotlib.pyplot as plt
# Note that "ops" in the comments refers "tensorflow operations". For the
# details: https://www.tensorflow.org/get_started/get_started

#Data directory
path = '/home/fzechini/Desktop/uie2/data/trainData_pp_rgb_masked.pkl'
data_split = utils.get_data(path)

# Model Hyperparameters
tf.flags.DEFINE_float("dropout_rate", 0.5, "Dropout rate (default: 0.5)")
# Training Parameters
tf.flags.DEFINE_integer("learning_rate", 5e-4, "Batch Size (default: 1e-3)")
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 32)")
tf.flags.DEFINE_integer("num_epochs", 10000, "Number of full passess over whole training data (default: 100)")
tf.flags.DEFINE_integer("print_every_step", 200, "Print training details after this many steps/iterations (i.e., batches) (default: 10)")
tf.flags.DEFINE_integer("evaluate_every_step", 1000, "Evaluate model on validation set after this many steps/iterations (i.e., batches) (default: 500)")
tf.flags.DEFINE_integer("checkpoint_every_step", 1000, "Save model after this many steps/iterations (i.e., batches) (default: 1000)")
tf.flags.DEFINE_string("log_dir", "./runs/", "Output directory (default: './runs/')")
tf.flags.DEFINE_integer("epoch_length", int(data_split[0].shape[0]/tf.flags.FLAGS.batch_size), "Batch Size (default: 1e-3)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nCommand-line Arguments:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Create a unique output directory for this experiment.
timestamp = str(int(time.time()))
FLAGS.model_dir = os.path.abspath(os.path.join(FLAGS.log_dir, timestamp))
print("Writing to {}\n".format(FLAGS.model_dir))

def main(unused_argv):
    # Load training data and crate a validation split.
    train_data = np.asarray(data_split[0], dtype=np.uint8)
    train_labels = np.asarray(data_split[1], dtype=np.int32)

    validation_data = np.asarray(data_split[2], dtype=np.uint8)
    validation_labels = np.asarray(data_split[3], dtype=np.int32)

    # Get input dimensionality.
    IMAGE_HEIGHT = train_data.shape[1]
    IMAGE_WIDTH = train_data.shape[2]
    NUM_CHANNELS = train_data.shape[3]

    # Placeholder variables are used to change the input to the graph.
    # This is where training samples and labels are fed to the graph.
    # These will be fed a batch of training data at each training step
    # using the {feed_dict} argument to the sess.run() call below.
    input_samples_op = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS], name="input_samples")
    input_label_op = tf.placeholder(tf.int32, shape=[None], name="input_labels")

    # Some layers/functions have different behaviours during training and evaluation.
    # If model is in the training mode, then pass True.
    mode = tf.placeholder(tf.bool, name="mode")
    # loss_avg and accuracy_avg will be used to update summaries externally.
    # Since we do evaluation by using batches, we may want average value.
    # (1) Keep counting number of correct predictions over batches.
    # (2) Calculate the average value, evaluate the corresponding summaries
    # by using loss_avg and accuracy_avg placeholders.
    loss_avg = tf.placeholder(tf.float32, name="loss_avg")
    accuracy_avg = tf.placeholder(tf.float32, name="accuracy_avg")
    learning_rate_1 = tf.placeholder(tf.float32, name="learning_rate_1")
    learning_rate_2 = tf.placeholder(tf.float32, name="learning_rate_2")

    # Call the function that builds the network. You should pass all the
    # parameters that controls external inputs.
    # It returns "logits" layer, i.e., the top-most layer of the network.
    logits = models.conv_model_with_layers_api(input_samples_op, FLAGS.dropout_rate, mode)
    
    # Optional:
    # Tensorflow provides a very simple and useful API (summary) for
    # monitoring the training via tensorboard
    # (https://www.tensorflow.org/get_started/summaries_and_tensorboard)
    # However, it is not trivial to visualize average accuracy over whole
    # dataset. Create two tensorflow variables in order to count number of
    # samples fed and correct predictions made. They are attached to
    # a summary op (see below).
    counter_correct_prediction = tf.Variable(0, name='counter_correct_prediction', trainable=False)
    counter_samples_fed = tf.Variable(0, name='counter_samples_fed', trainable=False)

    # Loss calculations: cross-entropy
    with tf.name_scope("cross_entropy_loss"):
        # Takes predictions of the network (logits) and ground-truth labels
        # (input_label_op), and calculates the cross-entropy loss.
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=input_label_op))


    # Accuracy calculations.
    with tf.name_scope("accuracy"):
        # Return list of predictions (useful for making a submission)
        predictions = tf.argmax(logits, 1, name="predictions")
        predictions = tf.Print(predictions,[predictions])
        # Return a bool tensor with shape [batch_size] that is true for the
        # correct predictions.
        correct_predictions = tf.nn.in_top_k(logits, input_label_op, 1)
        # Calculate the accuracy per minibatch.
        batch_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        # Number of correct predictions in order to calculate average accuracy afterwards.
        num_correct_predictions = tf.reduce_sum(tf.cast(correct_predictions, tf.int32))


    def do_evaluation(sess, samples, labels):
        '''
        Evaluation function.
        @param sess: tensorflow session object.
        @param samples: input data (numpy tensor)
        @param labels: ground-truth labels (numpy array)
        '''
        batches = utils.data_iterator(samples, labels, FLAGS.batch_size)
        # Keep track of this run.
        counter_accuracy = 0.0
        counter_loss = 0.0
        counter_batches = 0
        for batch_samples, batch_labels in batches:
            counter_batches += 1
            feed_dict = {input_samples_op: batch_samples,
                         input_label_op: batch_labels,
                         mode: False}
            results = sess.run([loss, num_correct_predictions], feed_dict=feed_dict)
            counter_loss += results[0]
            counter_accuracy += results[1]
        return (counter_loss/counter_batches, counter_accuracy/(counter_batches*FLAGS.batch_size))

    # Generate a variable to contain a counter for the global training step.
    # Note that it is useful if you save/restore your network.
    global_step = tf.Variable(1, name='global_step', trainable=False)

    # Create optimization op.
    with tf.name_scope('train'):
        learning_rate_1 = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                                   decay_steps=1*FLAGS.epoch_length, decay_rate=0.96, staircase=True)
        learning_rate_2 = tf.train.exponential_decay(learning_rate_1, global_step,
                                                   decay_steps=30*FLAGS.epoch_length, decay_rate=0.1, staircase=True)
        #optimizer = tf.train.RMSPropOptimizer(learning_rate, 0.9, momentum=0.9, epsilon=1.0)
        optimizer = tf.train.AdamOptimizer(learning_rate_2)
        #optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate_2, momentum = 0.9)
        #optimizer = tf.train.AdadeltaOptimizer()
        train_op = optimizer.minimize(loss, global_step=global_step)


    # For saving/restoring the model.
    # Save important ops (which can be required later!) by adding them into
    # the collection. We will use them in order to evaluate our model on the test
    # data after training.
    # See tf.get_collection for details.
    tf.add_to_collection('predictions', predictions)
    tf.add_to_collection('input_samples_op', input_samples_op)
    tf.add_to_collection('mode', mode)

    # Create session object
    sess = tf.Session()
    # Add the ops to initialize variables.
    init_op = tf.global_variables_initializer()
    # Actually intialize the variables
    sess.run(init_op)

    # Create summary ops for monitoring the training.
    # Each summary op annotates a node in the computational graph and collects
    # data data from it.
    summary_trian_loss = tf.summary.scalar('loss', loss)
    summary_train_acc = tf.summary.scalar('accuracy_training', batch_accuracy)
    summary_avg_accuracy = tf.summary.scalar('accuracy_avg', accuracy_avg)
    summary_avg_loss = tf.summary.scalar('loss_avg', loss_avg)
    summary_learning_rate = tf.summary.scalar('learning_rate', learning_rate_2)
    summary_images = tf.summary.image('images', input_samples_op, max_outputs=10)
    #summary_model = tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

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

    # Define counters in order to accumulate measurements.
    counter_correct_predictions_training = 0.0
    counter_loss_training = 0.0
    for epoch in range(1, FLAGS.num_epochs+1):
        # Generate training batches
        training_batches = utils.data_iterator(train_data, train_labels, FLAGS.batch_size)#, FLAGS.num_epochs)
        # Training loop.
        for batch_samples, batch_labels in training_batches:
            step = tf.train.global_step(sess, global_step)

            if (step%FLAGS.checkpoint_every_step) == 0:
                ckpt_save_path = saver.save(sess, os.path.join(FLAGS.model_dir, 'model'), global_step)
                print("Model saved in file: %s" % ckpt_save_path)

            # This dictionary maps the batch data (as a numpy array) to the
            # placeholder variables in the graph.
            feed_dict = {input_samples_op: batch_samples, # .eval(session=sess),
                         input_label_op: batch_labels,
                         mode: True}

            # Run the optimizer to update weights.
            # Note that "train_op" is responsible from updating network weights.
            # Only the operations that are fed are evaluated.
            # Run the optimizer to update weights.
            train_summary, correct_predictions_training, loss_training, _ = sess.run([summaries_training, num_correct_predictions, loss, train_op], feed_dict=feed_dict)
            # Update counters.
            counter_correct_predictions_training += correct_predictions_training
            counter_loss_training += loss_training
            # Write summary data.
            train_summary_writer.add_summary(train_summary, step)

            # Occasionally print status messages.
            if (step%FLAGS.print_every_step) == 0:
                # Calculate average training accuracy.
                accuracy_avg_value_training = counter_correct_predictions_training/(FLAGS.print_every_step*FLAGS.batch_size)
                loss_avg_value_training = counter_loss_training/(FLAGS.print_every_step)
                # [Epoch/Iteration]
                print("[%d/%d] [Training] Accuracy: %.3f, Loss: %.3f" % (epoch, step, accuracy_avg_value_training, loss_avg_value_training))
                counter_correct_predictions_training = 0.0
                counter_loss_training = 0.0
                # Report
                # Note that accuracy_avg and loss_avg placeholders are defined
                # just to feed average results to summaries.
                summary_report = sess.run(summaries_evaluation, feed_dict={accuracy_avg:accuracy_avg_value_training, loss_avg:loss_avg_value_training}) 
                train_summary_writer.add_summary(summary_report, step)

            if (step%FLAGS.evaluate_every_step) == 0:
                # Calculate average validation accuracy.
                (loss_avg_value_validation, accuracy_avg_value_validation) = do_evaluation(sess, validation_data, validation_labels)
                print("[%d/%d] [Validation] Accuracy: %.3f, Loss: %.3f" % (epoch, step, accuracy_avg_value_validation, loss_avg_value_validation))
                # Report
                summary_report = sess.run(summaries_evaluation, feed_dict={accuracy_avg:accuracy_avg_value_validation, loss_avg:loss_avg_value_validation})
                valid_summary_writer.add_summary(summary_report, step)

if __name__ == '__main__':
    tf.app.run()
