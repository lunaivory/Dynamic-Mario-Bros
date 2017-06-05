import time
import numpy as np
import tensorflow as tf
from constants import *

def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))

def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def _float_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_float_feature(v) for v in values])


def sparse_tuple_from(sequence, dtype=np.int32):
    indices = np.asarray(list(zip([0] * len(sequence), range(len(sequence)))), dtype=np.int64)
    values = np.asarray(sequence, dtype=dtype)
    shape = np.asarray([1, len(sequence)], dtype=np.int64)

    return indices, values, shape

# Function used to inspect data get from tfrecords files
# Example usage :
#   look_into_tfRecords(['%s/%s/Sample%04d.tfrecords' % (TFRecord_DATA_PATH, 'Train', i) for i in range(1,2)], 'Train')
#   look_into_tfRecords(['%s/%s/Sample%04d.tfrecords' % (TFRecord_DATA_PATH, 'Test', i) for i in range(701,703)], 'Test')
#   look_into_tfRecords(TRAIN_FILENAMES + VALIDATION_FILENAMES, 'Train')
# TODO : Fix it to current feature list
def look_into_tfRecords(filenames, data_type):
    #    get_ipython().magic(u'matplotlib inline')
    print(filenames)

    # input_samples_op, input_labels_op = input_pipeline(filenames, data_type)
    test = input_pipeline(filenames, data_type)
    '''input to ctc test'''
    # input_seq_op = tf.Variable([CLIPS_PER_VIDEO] * BATCH_SIZE)
    # mode = tf.placeholder(tf.bool, name='mode') # Pass True in when it is in the trainging mode
    # logits = models.conv_model_with_layers_api(input_samples_op, 0.5, mode)
    # loss = tf.nn.ctc_loss(input_labels_op, logits, input_seq_op, time_major=False)

    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # Create threads to prefetch the data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    data = sess.run([test])

    # Print
    print("# Samples: " + str(len(input_samples)))
    print("Sequence labels: " + str(input_labels))

    # Note that the second dimension will give maximum-length in the batch, i.e., the padded sequence length.
    print("Sequence type: " + str(type(input_samples)))
    print("Sequence shape: " + str(input_samples.shape))

    # Fetch first clips 11th frame.
    img = input_samples[0][50][0]
    print("Image shape: " + str(img.shape))


#    plt.figure()
#    plt.axis("off")
#    plt.imshow(img) # Note that image may look wierd because it is normalized.

# look_into_tfRecords(TRAIN_FILENAMES, 'Train')

'''######################################################'''
'''#        functions used in network creation          #'''
'''######################################################'''


def sparse_tuple_from(sequence, dtype=np.int32):
    indices = np.asarray(list(zip([0] * len(sequence), range(len(sequence)))), dtype=np.int64)
    values = np.asarray(sequence, dtype=dtype)
    shape = np.asarray([1, len(sequence)], dtype=np.int64)

    return indices, values, shape

    # TODO can we delete this?
    # def sparse_tuple_from(sequences, dtype=np.int32):
    #    """Create a sparse representention of x.
    #    Args:
    #        sequences: a list of lists of type dtype where each element is a sequence
    #    Returns:
    #        A tuple with (indices, values, shape)
    #    """
    #    indices = []
    #    values = []
    #
    #    for n, seq in enumerate(sequences):
    #        indices.extend(zip([n]*len(seq), range(len(seq))))
    #        values.extend(seq)
    #
    #    indices = np.asarray(indices, dtype=np.int64)
    #    values = np.asarray(values, dtype=np.int64)
    #    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)
    #
    #    return indices, values, shape


    # look_into_tfRecords(TRAIN_FILENAMES, 'Train')