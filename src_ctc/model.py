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
    weight_decay = WEIGHT_DECAY_CNN
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
            kernel_regularizer = slim.l2_regularizer(weight_decay),
            bias_regularizer = slim.l2_regularizer(weight_decay),
            name='conv1')
    # Pooling Layer #1

        pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[1, 2, 2], strides=[1,2,2], padding="same")

        # global norm per clip
        pool1_norm = [tf.nn.batch_normalization(i,
                                                          mean=tf.nn.moments(i, axes=[0,1,2])[0],
                                                          variance=tf.nn.moments(i, axes=[0,1,2])[1],
                                                          offset=None, scale=None, variance_epsilon=1e-10)
                                for i in tf.unstack(pool1)]
        pool1_norm = tf.stack(pool1_norm)

    # Convolutional Layer #2

        conv2 = tf.layers.conv3d(
            inputs =pool1_norm,
            filters = 128,
            kernel_size = [3, 3, 3],
            padding = "same",
            activation = tf.nn.relu,
            kernel_regularizer = slim.l2_regularizer(weight_decay),
            bias_regularizer = slim.l2_regularizer(weight_decay),
            name='conv2')

    # Pooling Layer #2

        pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=[2,2,2], padding="same")
        # global norm per clip
        pool2_norm = [tf.nn.batch_normalization(i,
                                                          mean=tf.nn.moments(i, axes=[0,1,2])[0],
                                                          variance=tf.nn.moments(i, axes=[0,1,2])[1],
                                                          offset=None, scale=None, variance_epsilon=1e-10)
                                for i in tf.unstack(pool2)]
        pool2_norm = tf.stack(pool2_norm)

    # Convolutional Layer #3

        conv3 = tf.layers.conv3d(
            inputs=pool2_norm,
            filters=256,
            kernel_size=[3, 3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_regularizer=slim.l2_regularizer(weight_decay),
            bias_regularizer=slim.l2_regularizer(weight_decay),
            name='conv3')

        # global norm per clip
        conv3_norm = [tf.nn.batch_normalization(i,
                                                          mean=tf.nn.moments(i, axes=[0,1,2])[0],
                                                          variance=tf.nn.moments(i, axes=[0,1,2])[1],
                                                          offset=None, scale=None, variance_epsilon=1e-10)
                                for i in tf.unstack(conv3)]
        conv3_norm = tf.stack(conv3_norm)

    # Convolutional Layer #4

        conv4 = tf.layers.conv3d(
            inputs=conv3_norm,
            filters=256,
            kernel_size=[3, 3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_regularizer=slim.l2_regularizer(weight_decay),
            bias_regularizer=slim.l2_regularizer(weight_decay),
            name='conv4'
        )

    # Pooling Layer #3

        pool3 = tf.layers.max_pooling3d(inputs=conv4, pool_size=[2, 2, 2], strides=[2,2,2], padding="same")
        
        # global norm per clip
        pool3_norm = [tf.nn.batch_normalization(i,
                                                          mean=tf.nn.moments(i, axes=[0,1,2])[0],
                                                          variance=tf.nn.moments(i, axes=[0,1,2])[1],
                                                          offset=None, scale=None, variance_epsilon=1e-10)
                                for i in tf.unstack(pool3)]
        pool3_norm = tf.stack(pool3_norm)

    # Convolutional Layer #5

        conv5 = tf.layers.conv3d(
            inputs=pool3_norm,
            filters=512,
            kernel_size=[3, 3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_regularizer=slim.l2_regularizer(weight_decay),
            bias_regularizer=slim.l2_regularizer(weight_decay),
            name='conv5'
        )
        
        # global norm per clip
        conv5_norm = [tf.nn.batch_normalization(i,
                                                          mean=tf.nn.moments(i, axes=[0,1,2])[0],
                                                          variance=tf.nn.moments(i, axes=[0,1,2])[1],
                                                          offset=None, scale=None, variance_epsilon=1e-10)
                                for i in tf.unstack(conv5)]
        conv5_norm = tf.stack(conv5_norm)

    # Convolutional Layer #6

        conv6 = tf.layers.conv3d(
            inputs=conv5_norm,
            filters=512,
            kernel_size=[3, 3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_regularizer=slim.l2_regularizer(weight_decay),
            bias_regularizer=slim.l2_regularizer(weight_decay),
            name='conv6'
        )

    # Pooling Layer #4

        pool4 = tf.layers.max_pooling3d(inputs=conv6, pool_size=[2, 2, 2], strides=[2,2,2], padding="same")
        
        # global norm per clip
        pool4_norm = [tf.nn.batch_normalization(i,
                                                          mean=tf.nn.moments(i, axes=[0,1,2])[0],
                                                          variance=tf.nn.moments(i, axes=[0,1,2])[1],
                                                          offset=None, scale=None, variance_epsilon=1e-10)
                                for i in tf.unstack(pool4)]
        pool4_norm = tf.stack(pool4_norm)

    # Convolutional Layer #7

        conv7 = tf.layers.conv3d(
            inputs=pool4_norm,
            filters=512,
            kernel_size=[3, 3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_regularizer=slim.l2_regularizer(weight_decay),
            bias_regularizer=slim.l2_regularizer(weight_decay),
            name='conv7'
        )

        # global norm per clip
        conv7_norm = [tf.nn.batch_normalization(i,
                                                          mean=tf.nn.moments(i, axes=[0,1,2])[0],
                                                          variance=tf.nn.moments(i, axes=[0,1,2])[1],
                                                          offset=None, scale=None, variance_epsilon=1e-10)
                                for i in tf.unstack(conv7)]
        conv7_norm = tf.stack(conv7_norm)

    # Convolutional Layer #8

        conv8 = tf.layers.conv3d(
            inputs=conv7_norm,
            filters=512,
            kernel_size=[3, 3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_regularizer=slim.l2_regularizer(weight_decay),
            bias_regularizer=slim.l2_regularizer(weight_decay),
            name='conv8'
        )
        
        # global norm per clip
        conv8_norm = [tf.nn.batch_normalization(i,
                                                          mean=tf.nn.moments(i, axes=[0,1,2])[0],
                                                          variance=tf.nn.moments(i, axes=[0,1,2])[1],
                                                          offset=None, scale=None, variance_epsilon=1e-10)
                                for i in tf.unstack(conv8)]
        conv8_norm = tf.stack(conv8_norm)

    # last max pooling no stride on temporal dim since we are using clips of 8 frames and here we reached dim 1, in
    # the paper they dont use this layer but I think we should add it otherwise tensor too large
    # Pooling Layer #5

        pool5 = tf.layers.max_pooling3d(inputs=conv8_norm, pool_size=[1, 2, 2], strides=[1, 2, 2], padding="same")

        # global norm per clip
        pool5_norm = [tf.nn.batch_normalization(i,
                                                          mean=tf.nn.moments(i, axes=[0,1,2])[0],
                                                          variance=tf.nn.moments(i, axes=[0,1,2])[1],
                                                          offset=None, scale=None, variance_epsilon=1e-10)
                                for i in tf.unstack(pool5)]
        pool5_norm = tf.stack(pool5_norm)

    # Flatten tensor into a batch of vectors
    # shape=[batch_size, max_time, flat

        pool5_flat = tf.reshape(pool5_norm, shape=[-1, 4 * 4 * 512])

    # Dense Layer

        dense1 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu,
                                 kernel_regularizer=slim.l2_regularizer(weight_decay),
                                 bias_regularizer=slim.l2_regularizer(weight_decay),
                                 name='dense1')

    # Add dropout operation

        dropout1 = tf.layers.dropout(inputs=dense1, rate=dropout_rate, training=mode)

    # Dense Layer

        dense2 = tf.layers.dense(inputs=dropout1, units=2048, activation=tf.nn.relu,
                                 kernel_regularizer=slim.l2_regularizer(weight_decay),
                                 bias_regularizer=slim.l2_regularizer(weight_decay),
                                 name='dense2')

    # Add dropout operation

        dropout2 = tf.layers.dropout(inputs=dense2, rate=dropout_rate, training=mode)
           
        return dropout2
