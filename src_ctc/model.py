import tensorflow as tf
from constants import *
slim = tf.contrib.slim

def get_scope_variable(scope_name, var, shape=None):
    with tf.variable_scope(scope_name) as scope:
        try:
            v = tf.get_variable(var, shape)
        except ValueError:
            scope.reuse_variables()
            v = tf.get_variable(var)
    return v

def dynamic_mario_bros(input_layer, dropout_rate, mode, reuse=False):
    """
    Input layer [num_clips(250), frames_per_clip(8), height, length, rgb]
    
    ... then conv/max_pool until pool5_flat ... -> Input to LSTM [1, 250, 51200]
    
    Output logits [batch_size, num_clips(250), num_classes+2]
    """
    weight_decay=0.0 #0.0001
    # TODO they dont mention any activations for the conv layers, to check
    with tf.variable_scope("3Dcnn_model", reuse=reuse, initializer=tf.contrib.layers.xavier_initializer()):
    #with tf.name_scope("3Dcnn_model"):
        # Convolutional Layer #1

        conv1 = tf.layers.conv3d(
            inputs =input_layer,
            filters = 64,
            kernel_size = [3, 3, 3],
            padding = "same",
            activation = tf.nn.relu,
            # kernel_regularizer = slim.l2_regularizer(weight_decay),
            # bias_regularizer = slim.l2_regularizer(weight_decay),
            name='conv1')
    # Pooling Layer #1

        pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[1, 2, 2], strides=[1,2,2], padding="same")

        pool1_norm = tf.layers.batch_normalization(
            pool1,
            axis=0,
            training=mode)


    # Convolutional Layer #2

        conv2 = tf.layers.conv3d(
            inputs =pool1_norm,
            filters = 128,
            kernel_size = [3, 3, 3],
            padding = "same",
            activation = tf.nn.relu,
            #kernel_regularizer = slim.l2_regularizer(weight_decay),
            #bias_regularizer = slim.l2_regularizer(weight_decay),
            name='conv2')

    # Pooling Layer #2

        pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=[2,2,2], padding="same")
        pool2_norm = tf.layers.batch_normalization(
            pool2,
            axis=0,
            training=mode)

    # Convolutional Layer #3

        conv3 = tf.layers.conv3d(
            inputs=pool2_norm,
            filters=256,
            kernel_size=[3, 3, 3],
            padding="same",
            activation=tf.nn.relu,
            #kernel_regularizer=slim.l2_regularizer(weight_decay),
            #bias_regularizer=slim.l2_regularizer(weight_decay),
            name='conv3')

        conv3_norm = tf.layers.batch_normalization(
            conv3,
            axis=0,
            training=mode)
    # Convolutional Layer #4

        conv4 = tf.layers.conv3d(
            inputs=conv3_norm,
            filters=256,
            kernel_size=[3, 3, 3],
            padding="same",
            activation=tf.nn.relu,
            #kernel_regularizer=slim.l2_regularizer(weight_decay),
            #bias_regularizer=slim.l2_regularizer(weight_decay),
            name='conv4'
        )

    # Pooling Layer #3

        pool3 = tf.layers.max_pooling3d(inputs=conv4, pool_size=[2, 2, 2], strides=[2,2,2], padding="same")
        pool3_norm = tf.layers.batch_normalization(
            pool3,
            axis=0,
            training=mode)

    # Convolutional Layer #5

        conv5 = tf.layers.conv3d(
            inputs=pool3_norm,
            filters=512,
            kernel_size=[3, 3, 3],
            padding="same",
            activation=tf.nn.relu,
            #kernel_regularizer=slim.l2_regularizer(weight_decay),
            #bias_regularizer=slim.l2_regularizer(weight_decay),
            name='conv5'
        )
        conv5_norm = tf.layers.batch_normalization(
            conv5,
            axis=0,
            training=mode)

    # Convolutional Layer #6

        conv6 = tf.layers.conv3d(
            inputs=conv5_norm,
            filters=512,
            kernel_size=[3, 3, 3],
            padding="same",
            activation=tf.nn.relu,
            #kernel_regularizer=slim.l2_regularizer(weight_decay),
            #bias_regularizer=slim.l2_regularizer(weight_decay),
            name='conv6'
        )

    # Pooling Layer #4

        pool4 = tf.layers.max_pooling3d(inputs=conv6, pool_size=[2, 2, 2], strides=[2,2,2], padding="same")
        pool4_norm = tf.layers.batch_normalization(
            pool4,
            axis=0,
            training=mode)

    # Convolutional Layer #7

        conv7 = tf.layers.conv3d(
            inputs=pool4_norm,
            filters=512,
            kernel_size=[3, 3, 3],
            padding="same",
            activation=tf.nn.relu,
            #kernel_regularizer=slim.l2_regularizer(weight_decay),
            #bias_regularizer=slim.l2_regularizer(weight_decay),
            name='conv7'
        )
        conv7_norm = tf.layers.batch_normalization(
            conv7,
            axis=0,
            training=mode)

    # Convolutional Layer #8

        conv8 = tf.layers.conv3d(
            inputs=conv7_norm,
            filters=512,
            kernel_size=[3, 3, 3],
            padding="same",
            activation=tf.nn.relu,
            #kernel_regularizer=slim.l2_regularizer(weight_decay),
            #bias_regularizer=slim.l2_regularizer(weight_decay),
            name='conv8'
        )
        conv8_norm = tf.layers.batch_normalization(
            conv8,
            axis=0,
            training=mode)

    # last max pooling no stride on temporal dim since we are using clips of 8 frames and here we reached dim 1, in
    # the paper they dont use this layer but I think we should add it otherwise tensor too large
    # Pooling Layer #5

        pool5 = tf.layers.max_pooling3d(inputs=conv8_norm, pool_size=[1, 2, 2], strides=[1, 2, 2], padding="same")

        pool5_norm = tf.layers.batch_normalization(
            pool5,
            axis=0,
            training=mode)

    # Flatten tensor into a batch of vectors
    # shape=[batch_size, max_time, flat

        pool5_flat = tf.reshape(pool5_norm, shape=[-1, 4 * 4 * 512])

    # Dense Layer

        dense1 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu,
                                 #kernel_regularizer=slim.l2_regularizer(weight_decay),
                                 #bias_regularizer=slim.l2_regularizer(weight_decay),
                                 name='dense1')

    # Add dropout operation

        dropout1 = tf.layers.dropout(inputs=dense1, rate=dropout_rate, training=mode)

    # Dense Layer

        dense2 = tf.layers.dense(inputs=dropout1, units=4096, activation=tf.nn.relu,
                                 #kernel_regularizer=slim.l2_regularizer(weight_decay),
                                 #bias_regularizer=slim.l2_regularizer(weight_decay),
                                 name='dense2')

    # Add dropout operation

        dropout2 = tf.layers.dropout(inputs=dense2, rate=dropout_rate, training=mode)
        dropout2_flat = tf.reshape(dropout2, shape=[1, -1, 4096])

        # with tf.name_scope("LSTM"):
        #     seq_length =CLIPS_PER_VIDEO # tf.shape(dropout2_flat)[1]
        #     lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=2048)
        #     lstm_outputs, _ = tf.nn.dynamic_rnn(lstm_cell, dropout2_flat, dtype=tf.float32, time_major=False, sequence_length=[seq_length])
        #
        # # Add dropout operation
        # with tf.name_scope("dropout3"):
        #     dropout3 = tf.layers.dropout(inputs=lstm_outputs, rate=dropout_rate, training=mode)
        #
        # # Dense Layer
        # with tf.name_scope("dense3"):
        #     dense3 = tf.layers.dense(inputs=dropout3, units=4096, activation=tf.nn.relu,
        #                              #kernel_regularizer=slim.l2_regularizer(weight_decay),
        #                              #bias_regularizer=slim.l2_regularizer(weight_decay)
        #                              )
        #
        # # Add dropout operation
        # with tf.name_scope("dropout4"):
        #     dropout4 = tf.layers.dropout(inputs=dense3, rate=dropout_rate, training=mode)
        #
        # with tf.name_scope("logits"):
        #     logits = tf.layers.dense(inputs=dropout4, units=21)
           
        return dropout2_flat
