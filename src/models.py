import tensorflow as tf
slim = tf.contrib.slim

def conv_model_with_layers_api(input_layer, dropout_rate, mode):
    """
    Builds a model by using tf.layers API.

    Note that in mnist_fc_with_summaries.ipynb weights and biases are
    defined manually. tf.layers API follows similar steps in the background.
    (you can check the difference between tf.nn.conv2d and tf.layers.conv2d)
    """
    weight_decay=0.005
    # TODO they dont mention any activations for the conv layers, to check
    with tf.name_scope("network"):

        # Convolutional Layer #1
        with tf.name_scope("cnn1"):
            conv1 = tf.layers.conv3d(
                inputs =input_layer,
                filters = 64,
                kernel_size = [3, 3, 3],
                padding = "same",
                activation = tf.nn.relu,
                kernel_regularizer = slim.l2_regularizer(weight_decay),
                bias_regularizer = slim.l2_regularizer(weight_decay),
                #bias_initializer = tf.constant_initializer(0.01)
                )

        # Pooling Layer #1
        with tf.name_scope("pooling1"):
            pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[1, 2, 2], strides=1, padding="same")

        # Convolutional Layer #2
        with tf.name_scope("cnn2"):
            conv2 = tf.layers.conv3d(
                inputs =pool1,
                filters = 128,
                kernel_size = [3, 3, 3],
                padding = "same",
                activation = tf.nn.relu,
                kernel_regularizer = slim.l2_regularizer(weight_decay),
                bias_regularizer = slim.l2_regularizer(weight_decay),
                #bias_initializer = tf.constant_initializer(0.01)
                )

        # Pooling Layer #2
        with tf.name_scope("pooling2"):
            pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=1, padding="same")

        # Convolutional Layer #3
        with tf.name_scope("cnn3"):
            conv3 = tf.layers.conv3d(
                inputs=pool2,
                filters=256,
                kernel_size=[3, 3, 3],
                padding="same",
                activation=tf.nn.relu,
                kernel_regularizer=slim.l2_regularizer(weight_decay),
                bias_regularizer=slim.l2_regularizer(weight_decay),
                # bias_initializer = tf.constant_initializer(0.01)
            )

        # Convolutional Layer #4
        with tf.name_scope("cnn4"):
            conv4 = tf.layers.conv3d(
                inputs=conv3,
                filters=256,
                kernel_size=[3, 3, 3],
                padding="same",
                activation=tf.nn.relu,
                # kernel_regularizer=slim.l2_regularizer(weight_decay),
                # bias_regularizer=slim.l2_regularizer(weight_decay),
                # bias_initializer = tf.constant_initializer(0.01)
            )

        # Pooling Layer #3
        with tf.name_scope("pooling3"):
            pool3 = tf.layers.max_pooling3d(inputs=conv4, pool_size=[2, 2, 2], strides=1, padding="same")

        # Convolutional Layer #5
        with tf.name_scope("cnn5"):
            conv5 = tf.layers.conv3d(
                inputs=pool3,
                filters=512,
                kernel_size=[3, 3, 3],
                padding="same",
                activation=tf.nn.relu,
                # kernel_regularizer=slim.l2_regularizer(weight_decay),
                # bias_regularizer=slim.l2_regularizer(weight_decay),
                # bias_initializer = tf.constant_initializer(0.01)
            )

        # Convolutional Layer #2
        with tf.name_scope("cnn6"):
            conv6 = tf.layers.conv3d(
                inputs=conv5,
                filters=512,
                kernel_size=[3, 3, 3],
                padding="same",
                activation=tf.nn.relu,
                # kernel_regularizer=slim.l2_regularizer(weight_decay),
                # bias_regularizer=slim.l2_regularizer(weight_decay),
                # bias_initializer = tf.constant_initializer(0.01)
            )

        # Pooling Layer #4
        with tf.name_scope("pooling4"):
            pool4 = tf.layers.max_pooling3d(inputs=conv6, pool_size=[2, 2, 2], strides=1, padding="same")

        # Convolutional Layer #7
        with tf.name_scope("cnn7"):
            conv7 = tf.layers.conv3d(
                inputs=pool4,
                filters=512,
                kernel_size=[3, 3, 3],
                padding="same",
                activation=tf.nn.relu,
                # kernel_regularizer=slim.l2_regularizer(weight_decay),
                # bias_regularizer=slim.l2_regularizer(weight_decay),
                # bias_initializer = tf.constant_initializer(0.01)
            )

        # Convolutional Layer #8
        with tf.name_scope("cnn8"):
            conv8 = tf.layers.conv3d(
                inputs=conv7,
                filters=512,
                kernel_size=[3, 3, 3],
                padding="same",
                activation=tf.nn.relu,
                # kernel_regularizer=slim.l2_regularizer(weight_decay),
                # bias_regularizer=slim.l2_regularizer(weight_decay),
                # bias_initializer = tf.constant_initializer(0.01)
            )

        # Pooling Layer #5
        with tf.name_scope("pooling5"):
            pool5 = tf.layers.max_pooling3d(inputs=conv8, pool_size=[2, 2, 2], strides=1, padding="same")

        # Flatten tensor into a batch of vectors
        with tf.name_scope("flatten"):
            pool5_flat = tf.reshape(pool5, [-1, 38 * 38 * 128])

        # Dense Layer
        with tf.name_scope("dense1"):
            dense1 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu,
                                     kernel_regularizer=slim.l2_regularizer(weight_decay),
                                     bias_regularizer=slim.l2_regularizer(weight_decay),
                                     #bias_initializer = tf.constant_initializer(0.01)
                                     )

        # Add dropout operation
        with tf.name_scope("dropout1"):
            dropout1 = tf.layers.dropout(inputs=dense1, rate=dropout_rate, training=mode)

        # Dense Layer
        with tf.name_scope("dense2"):
            dense2 = tf.layers.dense(inputs=dropout1, units=4096, activation=tf.nn.relu,
                                     kernel_regularizer=slim.l2_regularizer(weight_decay),
                                     bias_regularizer=slim.l2_regularizer(weight_decay),
                                     #bias_initializer = tf.constant_initializer(0.01)
                                     )

        # Add dropout operation
        with tf.name_scope("dropout2"):
            dropout2 = tf.layers.dropout(inputs=dense2, rate=dropout_rate, training=mode)

        # Add dropout operation
        with tf.name_scope("LSTM"):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=512)
            lstm_outputs, _ = tf.contrib.rnn.dynamic_rnn(lstm_cell, dropout2, dtype=tf.float32)
            # TODO add dropout wrapper for LSTM

        # Logits layer
        with tf.name_scope("logits"):
            logits = tf.layers.dense(inputs=lstm_outputs, units=20)

        return logits
