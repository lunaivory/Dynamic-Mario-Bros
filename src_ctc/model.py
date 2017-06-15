import tensorflow as tf
slim = tf.contrib.slim

def dynamic_mario_bros(input_layer, dropout_rate, mode, net_type):
    """
    Input layer [num_clips(250), frames_per_clip(8), height, length, rgb]
    
    ... then conv/max_pool until pool5_flat ... -> Input to LSTM [1, 250, 51200]
    
    Output logits [batch_size, num_clips(250), num_classes+2]
    """
    weight_decay=0.0001
    # TODO they dont mention any activations for the conv layers, to check
    with tf.name_scope("network"):

        with tf.variable_scope("norm_1", reuse=True):
            input_layer_norm = tf.layers.batch_normalization(
                                      input_layer,
                                      axis=0,
                                      training=mode)
        # Convolutional Layer #1
        with tf.variable_scope("cnn1", reuse=True):
            conv1 = tf.layers.conv3d(
                inputs =input_layer_norm,
                filters = 64,
                kernel_size = [3, 3, 3],
                padding = "same",
                activation = tf.nn.relu,
                #kernel_regularizer = slim.l2_regularizer(weight_decay),
                #bias_regularizer = slim.l2_regularizer(weight_decay)
                )

        # Pooling Layer #1
        with tf.variable_scope("pooling1", reuse=True):
            pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[1, 2, 2], strides=[1,2,2], padding="same")

        with tf.variable_scope("norm_9", reuse=True):
            pool1_norm = tf.layers.batch_normalization(
                                      pool1,
                                      axis=0,
                                      training=mode)

        # Convolutional Layer #2
        with tf.variable_scope("cnn2", reuse=True):
            conv2 = tf.layers.conv3d(
                inputs =pool1_norm,
                filters = 128,
                kernel_size = [3, 3, 3],
                padding = "same",
                activation = tf.nn.relu,
                #kernel_regularizer = slim.l2_regularizer(weight_decay),
                #bias_regularizer = slim.l2_regularizer(weight_decay)
                )

        # Pooling Layer #2
        with tf.variable_scope("pooling2", reuse=True):
            pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=[2,2,2], padding="same")

        with tf.variable_scope("norm_2", reuse=True):
            pool2_norm = tf.layers.batch_normalization(
                                      pool2,
                                      axis=0,
                                      training=mode)

        # Convolutional Layer #3
        with tf.variable_scope("cnn3", reuse=True):
            conv3 = tf.layers.conv3d(
                inputs=pool2_norm,
                filters=256,
                kernel_size=[3, 3, 3],
                padding="same",
                activation=tf.nn.relu,
                #kernel_regularizer=slim.l2_regularizer(weight_decay),
                #bias_regularizer=slim.l2_regularizer(weight_decay)
            )

        with tf.variable_scope("norm_3", reuse=True):
            conv3_norm = tf.layers.batch_normalization(
                                      conv3,
                                      axis=0,
                                      training=mode)
        # Convolutional Layer #4
        with tf.variable_scope("cnn4", reuse=True):
            conv4 = tf.layers.conv3d(
                inputs=conv3_norm,
                filters=256,
                kernel_size=[3, 3, 3],
                padding="same",
                activation=tf.nn.relu,
                #kernel_regularizer=slim.l2_regularizer(weight_decay),
                #bias_regularizer=slim.l2_regularizer(weight_decay),
            )

        # Pooling Layer #3
        with tf.variable_scope("pooling3", reuse=True):
            pool3 = tf.layers.max_pooling3d(inputs=conv4, pool_size=[2, 2, 2], strides=[2,2,2], padding="same")

        with tf.variable_scope("norm_4"):
            pool3_norm = tf.layers.batch_normalization(
                                      pool3,
                                      axis=0,
                                      training=mode)

        # Convolutional Layer #5
        with tf.variable_scope("cnn5", reuse=True):
            conv5 = tf.layers.conv3d(
                inputs=pool3_norm,
                filters=512,
                kernel_size=[3, 3, 3],
                padding="same",
                activation=tf.nn.relu,
                #kernel_regularizer=slim.l2_regularizer(weight_decay),
                #bias_regularizer=slim.l2_regularizer(weight_decay)
            )

        with tf.variable_scope("norm_5", reuse=True):
            conv5_norm = tf.layers.batch_normalization(
                                      conv5,
                                      axis=0,
                                      training=mode)

        # Convolutional Layer #6
        with tf.variable_scope("cnn6", reuse=True):
            conv6 = tf.layers.conv3d(
                inputs=conv5_norm,
                filters=512,
                kernel_size=[3, 3, 3],
                padding="same",
                activation=tf.nn.relu,
                #kernel_regularizer=slim.l2_regularizer(weight_decay),
                #bias_regularizer=slim.l2_regularizer(weight_decay)
            )

        # Pooling Layer #4
        with tf.variable_scope("pooling4", reuse=True):
            pool4 = tf.layers.max_pooling3d(inputs=conv6, pool_size=[2, 2, 2], strides=[2,2,2], padding="same")

        with tf.variable_scope("norm_6"):
            pool4_norm = tf.layers.batch_normalization(
                                      pool4,
                                      axis=0,
                                      training=mode)

        # Convolutional Layer #7
        with tf.variable_scope("cnn7", reuse=True):
            conv7 = tf.layers.conv3d(
                inputs=pool4_norm,
                filters=512,
                kernel_size=[3, 3, 3],
                padding="same",
                activation=tf.nn.relu,
                #kernel_regularizer=slim.l2_regularizer(weight_decay),
                #bias_regularizer=slim.l2_regularizer(weight_decay)
            )

        with tf.variable_scope("norm_7", reuse=True):
            conv7_norm = tf.layers.batch_normalization(
                                      conv7,
                                      axis=0,
                                      training=mode)

        # Convolutional Layer #8
        with tf.variable_scope("cnn8", reuse=True):
            conv8 = tf.layers.conv3d(
                inputs=conv7_norm,
                filters=512,
                kernel_size=[3, 3, 3],
                padding="same",
                activation=tf.nn.relu,
                #kernel_regularizer=slim.l2_regularizer(weight_decay),
                #bias_regularizer=slim.l2_regularizer(weight_decay)
            )

        # last max pooling no stride on temporal dim since we are using clips of 8 frames and here we reached dim 1, in
        # the paper they dont use this layer but I think we should add it otherwise tensor too large
        # Pooling Layer #5
        with tf.variable_scope("pooling5", reuse=True):
            pool5 = tf.layers.max_pooling3d(inputs=conv8, pool_size=[1, 2, 2], strides=[1, 2, 2], padding="same")

        with tf.variable_scope("norm_8", reuse=True):
            pool5_norm = tf.layers.batch_normalization(
                                      pool5,
                                      axis=0,
                                      training=mode)

        # Flatten tensor into a batch of vectors
        # shape=[batch_size, max_time, flat
        with tf.variable_scope("flatten", reuse=True):
            pool5_flat = tf.reshape(pool5_norm, shape=[-1, 4 * 4 * 512])

        # Dense Layer
        with tf.variable_scope("dense1", reuse=True):
            dense1 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu,
                                     #kernel_regularizer=slim.l2_regularizer(weight_decay),
                                     #bias_regularizer=slim.l2_regularizer(weight_decay)
                                     )

        # Add dropout operation
        with tf.variable_scope("dropout1", reuse=True):
            dropout1 = tf.layers.dropout(inputs=dense1, rate=dropout_rate, training=mode)

        # Dense Layer
        with tf.variable_scope("dense2", reuse=True):
            dense2 = tf.layers.dense(inputs=dropout1, units=4096, activation=tf.nn.relu,
                                     #kernel_regularizer=slim.l2_regularizer(weight_decay),
                                     #bias_regularizer=slim.l2_regularizer(weight_decay)
                                     )

        # Add dropout operation
        with tf.variable_scope("dropout2", reuse=True):
            dropout2 = tf.layers.dropout(inputs=dense2, rate=dropout_rate, training=mode)
            dropout2_flat = tf.reshape(dropout2, shape=[1, -1, 4096])

        with tf.variable_scope("LSTM", reuse=True):
            seq_length = tf.shape(dropout2_flat)[1]
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=512)
            lstm_outputs, _ = tf.nn.dynamic_rnn(lstm_cell, dropout2_flat, dtype=tf.float32, time_major=False, sequence_length=[seq_length])

        with tf.variable_scope("logits", reuse=True):
            logits_3dcnn = tf.layers.dense(inputs=dropout2, units=21)
            logits_ctc = tf.layers.dense(inputs=lstm_outputs, units=21)
           
        return logits_3dcnn, logits_ctc
