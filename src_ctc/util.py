import time
import numpy as np
import tensorflow as tf

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



'''#####################'''
'''# Input Pipeline    #'''
'''#####################'''

def preprocessing_op(image_op):
    '''most preprocess should be done while making the .tfrecords files?'''
    with tf.name_scope('preprocessing'):
        # Reshape serialized image
        image_op = tf.reshape(image_op, [FRAMES_PER_CLIP] + list(IMAGE_SIZE))
        # Integer to float
        image_op = tf.to_float(image_op)
        # Normalize (Zero-mean unit-variance) on single image
        image_op = tf.map_fn(lambda img: tf.image.per_image_standardization(img), image_op)
        
        return image_op
    
def read_and_decode(filename_queue):
    readerOptions = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)
    reader = tf.TFRecordReader(options=readerOptions)
    _, serialized_example = reader.read(filename_queue)
    
    with tf.name_scope('TFRecordDecoding'):
        context_encoded, features_encoded = tf.parse_single_sequence_example(
            serialized_example,
            context_features={'sample_id': tf.FixedLenFeature([], dtype=tf.int64)},
            sequence_features={'rgbs':tf.FixedLenSequenceFeature([],dtype=tf.string), # TODO: check string or bytes
                     'labels':tf.FixedLenSequenceFeature([], dtype=tf.int64)} # TODO: check string or bytes
        )
        seq_rgb = tf.decode_raw(features_encoded['rgbs'], tf.uint8)
        seq_label = features_encoded['labels']
        # apply preprocessing to single image using map_fn
        seq_rgb = tf.map_fn(lambda x: preprocessing_op(x), elems = seq_rgb, dtype=tf.float32, back_prop=False)
        
        return [seq_rgb, seq_label]
    

def input_pipeline(filenames, data_type):
    
    with tf.name_scope('input_pipeline'):
        # Create a input file queue
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=NUM_EPOCHS, shuffle=True)
        
        # Read data from .tfreords files and decode to a list of samples (Using threads)
        samples = [read_and_decode(filename_queue) for _ in range(NUM_READ_THREADS)]
        
        # Create batches
        # batch_join is used for N threads, can use batch_join_shuffle too
        batch_rgb, batch_labels = tf.train.batch_join(samples, 
                                                      batch_size = BATCH_SIZE,
                                                      capacity = QUEUE_CAPACITY,
                                                      shapes = [[CLIPS_PER_VIDEO, FRAMES_PER_CLIP] + list(IMAGE_SIZE), [CLIPS_PER_VIDEO, ]],
                                                      enqueue_many= False, 
                                                      dynamic_pad = False, 
                                                      name = 'batch_join')
        if (data_type == 'Train' or data_type == 'Validation'):
            return batch_rgb, batch_labels
        else:
            return batch_rgb

# Function used to inspect data get from tfrecords files
# Example usage :
#   look_into_tfRecords(['%s/%s/Sample%04d.tfrecords' % (TFRecord_DATA_PATH, 'Train', i) for i in range(1,2)], 'Train')
#   look_into_tfRecords(['%s/%s/Sample%04d.tfrecords' % (TFRecord_DATA_PATH, 'Test', i) for i in range(701,703)], 'Test')
#   look_into_tfRecords(TRAIN_FILENAMES + VALIDATION_FILENAMES, 'Train')
def look_into_tfRecords(filenames, data_type):
    get_ipython().magic(u'matplotlib inline')
    print(filenames)

    batch_samples_op, batch_labels_op = input_pipeline(filenames, data_type)

    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # Create threads to prefetch the data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    batch_samples, batch_labels = sess.run([batch_samples_op, batch_labels_op])
    
    # Print 
    print("# Samples: " + str(len(batch_samples)))
    print("Sequence labels: " + str(batch_labels))

    # Note that the second dimension will give maximum-length in the batch, i.e., the padded sequence length.
    print("Sequence type: " + str(type(batch_samples)))
    print("Sequence shape: " + str(batch_samples.shape))

    # Fetch first clips 11th frame.
    img = batch_samples[0][50][0]
    print("Image shape: " + str(img.shape))

    plt.figure()
    plt.axis("off")
    plt.imshow(img) # Note that image may look wierd because it is normalized.

