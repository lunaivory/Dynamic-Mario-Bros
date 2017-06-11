import time
import numpy as np
import tensorflow as tf
from constants import *

# def _int64_feature(value):
#     """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#
# def _float_feature(value):
#     """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
#     return tf.train.Feature(int64_list=tf.train.FloatList(value=[value]))
#
# def _bytes_feature(value):
#     """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))
#
# def _int64_feature_list(values):
#     """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
#     return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])
#
# def _bytes_feature_list(values):
#     """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
#     return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])
#
# def _float_feature_list(values):
#     """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
#     return tf.train.FeatureList(feature=[_float_feature(v) for v in values])



'''######################################################'''
'''#                   Input Pipeline                   #'''
'''######################################################'''

def video_preprocessing_training_op(video_op):
    with tf.name_scope('Single_gesture_video_preprocessing_training'):

        #define constant tensors needed for preprocessing
        clips_in_video = tf.constant(CROP[0], shape=[1], dtype=tf.int32)
        channels = tf.constant(CROP[3], shape=[1], dtype=tf.int32)

        # Reshape for easier preprocessing
        video_op = tf.cast(video_op, dtype=tf.float32)
        clip_op = tf.reshape(video_op, [CLIPS_PER_VIDEO * FRAMES_PER_CLIP] + list(IMAGE_SIZE))

        #### Take random crop of dimension CROP=(CLIPS_PER_VIDEO * FRAMES_PER_CLIP, 112, 112, 3)
        zero_tensor = tf.zeros(shape=[1], dtype=tf.int32)
        #col_crop_idx = tf.random_uniform(shape=[1],minval=0, maxval=(IMAGE_SIZE[0] - CROP[1]), dtype=tf.int32)
        #row_crop_idx = tf.random_uniform(shape=[1],minval=0, maxval=(IMAGE_SIZE[1] - CROP[2]), dtype=tf.int32)
        col_crop_idx = tf.constant(int((IMAGE_SIZE[0] - CROP[1])/2), shape=[1], dtype=tf.int32)
        row_crop_idx = tf.constant(int((IMAGE_SIZE[1] - CROP[2])/2), shape=[1], dtype=tf.int32)
        begin_crop = tf.squeeze(tf.stack([zero_tensor, col_crop_idx, row_crop_idx, zero_tensor]))
        processed_video = tf.slice(clip_op, begin=begin_crop, size=CROP)

        #### Random rotation of +- 15 deg
        angle = tf.random_uniform(shape=[1],minval=-ROT_ANGLE, maxval=ROT_ANGLE, dtype=tf.float32)
        processed_video = tf.contrib.image.rotate(processed_video, angles=angle)

        #### Random scaling
        ## do this by taking a crop of random size and then resizing to the original shape
        #begin_col = tf.random_uniform(shape=[1], minval=0, maxval=(int(0.2 * CROP[1])), dtype=tf.int32)
        #begin_row = tf.random_uniform(shape=[1], minval=0, maxval=(int(0.2 * CROP[2])), dtype=tf.int32)
        ## get crop window size scaling_col, scaling_row
        #crop_col = tf.constant(CROP[1], dtype=tf.int32)
        #crop_row = tf.constant(CROP[2], dtype=tf.int32)
        #scaling_col = tf.subtract(crop_col, begin_col)
        #scaling_row = tf.subtract(crop_row, begin_row)
        ## do scaling by slicing and then resizing to orignal size
        #begin_scaling = tf.squeeze(tf.stack([zero_tensor, begin_col, begin_row, zero_tensor]))
        #size_scaling = tf.squeeze(tf.stack([clips_in_video, scaling_col, scaling_row, channels]))
        #scaling_crop = tf.slice(processed_video, begin=begin_scaling, size=size_scaling)
        #processed_video = tf.image.resize_images(scaling_crop, size=[CROP[1], CROP[2]])
             
        #rand = tf.random_uniform(minval=0, maxval=1, shape=[], dtype=tf.float32)
        #const_prob = tf.constant(0.5, dtype=tf.float32)
        #processed_video = tf.case([(tf.less(rand, const_prob), lambda :processed_video)], default=lambda : tf.reverse(processed_video, axis=[2]))

        # reshape to correct size for nework
        processed_video = tf.reshape(processed_video, [CLIPS_PER_VIDEO, FRAMES_PER_CLIP, CROP[1], CROP[2], CROP[3]])
        
        # normalise single images
        #processed_video = tf.reshape(processed_video, [CLIPS_PER_VIDEO * FRAMES_PER_CLIP, CROP[1], CROP[2], CROP[3]])
        #mean, var = tf.nn.moments(processed_video, axes=[0,1,2,3])
        #processed_video = tf.nn.batch_normalization(processed_video, mean=mean, variance=var, offset=None, scale=None, variance_epsilon=1e-10)

    return processed_video
    
def read_and_decode(filename_queue):
    readerOptions = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)
    reader = tf.TFRecordReader(options=readerOptions, name='training_tfreader')
    _, serialized_example = reader.read(filename_queue)

    with tf.name_scope('TFRecordDecoding'):
        _, features_encoded = tf.parse_single_sequence_example(
            serialized_example,
            sequence_features={
                'rgbs':tf.FixedLenSequenceFeature([],dtype=tf.string),
                'label':tf.FixedLenSequenceFeature([],dtype=tf.string),
                'dense_label': tf.FixedLenSequenceFeature([], dtype=tf.string),
                'clip_label': tf.FixedLenSequenceFeature([], dtype=tf.string),
                'sample_id': tf.FixedLenSequenceFeature([], dtype=tf.string),
                'num_frames': tf.FixedLenSequenceFeature([], dtype=tf.string)
            }
        )

        # decode video and apply preprocessing to entire video
        seq_rgb = tf.decode_raw(features_encoded['rgbs'], tf.uint8)
        # seq_rgb_pp = video_preprocessing_training_op(seq_rgb)
        seq_rgb_pp = video_preprocessing_training_op(seq_rgb)

        # decode label and reshape it correct size for shuffle_batch
        seq_label = tf.decode_raw(features_encoded['label'], tf.int32)
        # seq_label = tf.reshape(seq_label, [1])
        seq_label = tf.reshape(seq_label, [1])

        # get dense labels to calculate accuracy
        seq_label_dense = tf.decode_raw(features_encoded['dense_label'], tf.int32)
        seq_label_dense = tf.reshape(seq_label_dense, [FRAMES_PER_VIDEO])

        # get per clip labels to train 3dcnn
        seq_label_clip = tf.decode_raw(features_encoded['clip_label'], tf.int32)
        seq_label_clip = tf.reshape(seq_label_clip, [CLIPS_PER_VIDEO])

        return seq_rgb_pp, seq_label, seq_label_dense, seq_label_clip

def input_pipeline(filenames):
    with tf.name_scope('input_pipeline_training'):

        # Create a input file queue
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=NUM_EPOCHS, shuffle=True, capacity=1000,
                                                        name='Training_string_input')

        # Read data from .tfrecords files and decode to a list of samples (Using threads)
        data, label, dense_label, clip_label = read_and_decode(filename_queue)

        # Create batches
        batch_rgb, batch_label, batch_dense_label, batch_clip_label = tf.train.shuffle_batch([data, label, dense_label, clip_label],
                                                        batch_size=BATCH_SIZE,
                                                        capacity=QUEUE_CAPACITY,
                                                        min_after_dequeue=int(QUEUE_CAPACITY / 2),
                                                       )

        #reshape video to correct shape
        batch_rgb = tf.reshape(batch_rgb, [BATCH_SIZE * CLIPS_PER_VIDEO, FRAMES_PER_CLIP, CROP[1], CROP[2], CROP[3]])

        # make sparse tensor from batch labels
        # values are zero for testing
        batch_label = tf.transpose(batch_label, perm=[1, 0])
        idx = tf.where(tf.not_equal(batch_label, 0))
        vals = tf.gather_nd(batch_label, idx)
        vals = tf.cast(vals, dtype=tf.int32)
        batch_label_sparse = tf.SparseTensor(idx, vals, batch_label.get_shape())

        # reshape dense labels (frame by frame annotations)
        batch_dense_label = tf.reshape(batch_dense_label, [BATCH_SIZE*FRAMES_PER_VIDEO])

        # reshape clip labels (frame by frame annotations)
        batch_clip_label = tf.reshape(batch_clip_label, [BATCH_SIZE*CLIPS_PER_VIDEO])

        return batch_rgb, batch_label_sparse, batch_dense_label, batch_clip_label



        #
        # if (data_type == 'Train' or data_type == 'Validation'):
        #     return batch_rgb, batch_label_sparse
        # else:
        #     return batch_rgb

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
    #input_seq_op = tf.Variable([CLIPS_PER_VIDEO] * BATCH_SIZE)
    #mode = tf.placeholder(tf.bool, name='mode') # Pass True in when it is in the trainging mode
    #logits = models.conv_model_with_layers_api(input_samples_op, 0.5, mode)
    #loss = tf.nn.ctc_loss(input_labels_op, logits, input_seq_op, time_major=False)

    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU':0}))
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
    
#look_into_tfRecords(TRAIN_FILENAMES, 'Train')

'''######################################################'''
'''#        functions used in network creation          #'''
'''######################################################'''

def sparse_tuple_from(sequence, dtype=np.int32):
    
    indices = np.asarray(list(zip([0] * len(sequence), range(len(sequence)))), dtype=np.int64)
    values = np.asarray(sequence, dtype=dtype)
    shape = np.asarray([1, len(sequence)], dtype=np.int64)

    return indices, values, shape

# TODO can we delete this?
#def sparse_tuple_from(sequences, dtype=np.int32):
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
