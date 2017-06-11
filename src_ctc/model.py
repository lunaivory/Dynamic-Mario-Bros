import tensorflow as tf
slim = tf.contrib.slim

def dynamic_mario_bros(input_layer, dropout_rate, mode):
    """
    Input layer [num_clips(250), frames_per_clip(8), height, length, rgb]
    
    ... then conv/max_pool until pool5_flat ... -> Input to LSTM [1, 250, 51200]
    
    Output logits [batch_size, num_clips(250), num_classes+2]
    """
    weight_decay=0.0001
    # TODO they dont mention any activations for the conv layers, to check
    with tf.name_scope("network"): 
        
        #with tf.name_scope("norm_1"): 
        #    input_layer_norm = tf.layers.batch_normalization(
        #                              input_layer,
        #                              axis=0,
        #                              training=mode)
        # Convolutional Layer #1
        with tf.name_scope("cnn1"):
            conv1 = tf.layers.conv3d(
                inputs =input_layer,
                filters = 64,
                kernel_size = [3, 3, 3],
                padding = "same",
                activation = tf.nn.relu,
                #kernel_regularizer = slim.l2_regularizer(weight_decay),
                #bias_regularizer = slim.l2_regularizer(weight_decay)
                )

        # Pooling Layer #1
        with tf.name_scope("pooling1"):
            pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[1, 2, 2], strides=[1,2,2], padding="same")

        #with tf.name_scope("norm_9"):
        #    pool1_norm = tf.layers.batch_normalization(
        #                              pool1,
        #                              axis=0,
        #                              training=mode)

        # Convolutional Layer #2
        with tf.name_scope("cnn2"):
            conv2 = tf.layers.conv3d(
                inputs =pool1,
                filters = 128,
                kernel_size = [3, 3, 3],
                padding = "same",
                activation = tf.nn.relu,
                #kernel_regularizer = slim.l2_regularizer(weight_decay),
                #bias_regularizer = slim.l2_regularizer(weight_decay)
                )

        # Pooling Layer #2
        with tf.name_scope("pooling2"):
            pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=[2,2,2], padding="same")

        #with tf.name_scope("norm_2"):
        #    pool2_norm = tf.layers.batch_normalization(
        #                              pool2,
        #                              axis=0,
        #                              training=mode)

        # Convolutional Layer #3
        with tf.name_scope("cnn3"):
            conv3 = tf.layers.conv3d(
                inputs=pool2,
                filters=256,
                kernel_size=[3, 3, 3],
                padding="same",
                activation=tf.nn.relu,
                #kernel_regularizer=slim.l2_regularizer(weight_decay),
                #bias_regularizer=slim.l2_regularizer(weight_decay)
            )
        
        #with tf.name_scope("norm_3"):
        #    conv3_norm = tf.layers.batch_normalization(
        #                              conv3,
        #                              axis=0,
        #                              training=mode)
        # Convolutional Layer #4
        with tf.name_scope("cnn4"):
            conv4 = tf.layers.conv3d(
                inputs=conv3,
                filters=256,
                kernel_size=[3, 3, 3],
                padding="same",
                activation=tf.nn.relu,
                #kernel_regularizer=slim.l2_regularizer(weight_decay),
                #bias_regularizer=slim.l2_regularizer(weight_decay),
            )

        # Pooling Layer #3
        with tf.name_scope("pooling3"):
            pool3 = tf.layers.max_pooling3d(inputs=conv4, pool_size=[2, 2, 2], strides=[2,2,2], padding="same")

        #with tf.name_scope("norm_4"):
        #    pool3_norm = tf.layers.batch_normalization(
        #                              pool3,
        #                              axis=0,
        #                              training=mode)

        # Convolutional Layer #5
        with tf.name_scope("cnn5"):
            conv5 = tf.layers.conv3d(
                inputs=pool3,
                filters=512,
                kernel_size=[3, 3, 3],
                padding="same",
                activation=tf.nn.relu,
                #kernel_regularizer=slim.l2_regularizer(weight_decay),
                #bias_regularizer=slim.l2_regularizer(weight_decay)
            )
        
        #with tf.name_scope("norm_5"):
        #    conv5_norm = tf.layers.batch_normalization(
        #                              conv5,
        #                              axis=0,
        #                              training=mode)

        # Convolutional Layer #6
        with tf.name_scope("cnn6"):
            conv6 = tf.layers.conv3d(
                inputs=conv5,
                filters=512,
                kernel_size=[3, 3, 3],
                padding="same",
                activation=tf.nn.relu,
                #kernel_regularizer=slim.l2_regularizer(weight_decay),
                #bias_regularizer=slim.l2_regularizer(weight_decay)
            )

        # Pooling Layer #4
        with tf.name_scope("pooling4"):
            pool4 = tf.layers.max_pooling3d(inputs=conv6, pool_size=[2, 2, 2], strides=[2,2,2], padding="same")

        #with tf.name_scope("norm_6"):
        #    pool4_norm = tf.layers.batch_normalization(
        #                              pool4,
        #                              axis=0,
        #                              training=mode)

        # Convolutional Layer #7
        with tf.name_scope("cnn7"):
            conv7 = tf.layers.conv3d(
                inputs=pool4,
                filters=512,
                kernel_size=[3, 3, 3],
                padding="same",
                activation=tf.nn.relu,
                #kernel_regularizer=slim.l2_regularizer(weight_decay),
                #bias_regularizer=slim.l2_regularizer(weight_decay)
            )
        
        #with tf.name_scope("norm_7"):
        #    conv7_norm = tf.layers.batch_normalization(
        #                              conv7,
        #                              axis=0,
        #                              training=mode)

        # Convolutional Layer #8
        with tf.name_scope("cnn8"):
            conv8 = tf.layers.conv3d(
                inputs=conv7,
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
        with tf.name_scope("pooling5"):
            pool5 = tf.layers.max_pooling3d(inputs=conv8, pool_size=[1, 2, 2], strides=[1, 2, 2], padding="same")
        
        with tf.name_scope("norm_8"): 
            pool5_norm = tf.layers.batch_normalization(
                                      pool5,
                                      axis=0,
                                      training=mode)

        # Flatten tensor into a batch of vectors
        # shape=[batch_size, max_time, flat
        with tf.name_scope("flatten"):
            pool5_flat = tf.reshape(pool5_norm, shape=[-1, 4 * 4 * 512])

        # Dense Layer
        with tf.name_scope("dense1"):
            dense1 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu,
                                     #kernel_regularizer=slim.l2_regularizer(weight_decay),
                                     #bias_regularizer=slim.l2_regularizer(weight_decay)
                                     )

        # Add dropout operation
        with tf.name_scope("dropout1"):
            dropout1 = tf.layers.dropout(inputs=dense1, rate=dropout_rate, training=mode)

        
        #with tf.name_scope("norm_9"):
        #    dropout1_norm = tf.layers.batch_normalization(
        #                              dropout1,
        #                              axis=0,
        #                              training=mode)

        # Dense Layer
        with tf.name_scope("dense2"):
            dense2 = tf.layers.dense(inputs=dropout1, units=4096, activation=tf.nn.relu,
                                     #kernel_regularizer=slim.l2_regularizer(weight_decay),
                                     #bias_regularizer=slim.l2_regularizer(weight_decay)
                                     )
        
        # Add dropout operation
        with tf.name_scope("dropout2"):
            dropout2 = tf.layers.dropout(inputs=dense2, rate=dropout_rate, training=mode)
            #dropout2_flat = tf.reshape(dropout2, shape=[1, -1, 4096])
       
        #with tf.name_scope("norm_10"):
        #    dropout2_norm = tf.layers.batch_normalization(
        #                              dropout2_flat,
        #                              axis=0,
        #                              training=mode)
 
        #with tf.name_scope("LSTM"):
            #seq_length = tf.shape(dropout2_norm)[1]
            #lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=512)
            #lstm_outputs, _ = tf.nn.dynamic_rnn(lstm_cell, dropout2_norm, dtype=tf.float32, time_major=False, sequence_length=[seq_length])
            # TODO add dropout wrapper for LSTMs

        # Logits layer
        with tf.name_scope("logits"):
            logits = tf.layers.dense(inputs=dropout2, units=21)

        return logits
