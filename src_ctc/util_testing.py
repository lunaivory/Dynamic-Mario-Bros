import time
import numpy as np
import tensorflow as tf
from constants import *


'''######################################################'''
'''#                   Input Pipeline                   #'''
'''######################################################'''

def video_preprocessing_testing_op(video_op):
    with tf.name_scope('Single_gesture_video_preprocessing_testing'):

        #define constant tensors needed for preprocessing
        clips_in_video = tf.constant(CROP[0], shape=[1], dtype=tf.int32)
        channels = tf.constant(CROP[3], shape=[1], dtype=tf.int32)

        # Reshape for easier preprocessing
        video_op = tf.cast(video_op, dtype=tf.float32)
        clip_op = tf.reshape(video_op, [CLIPS_PER_VIDEO * FRAMES_PER_CLIP] + list(IMAGE_SIZE))

        # Create center crop
        zero_tensor = tf.zeros(shape=[1], dtype=tf.int32)
        col_crop_idx = tf.constant(int((IMAGE_SIZE[0] - CROP[1])/2), shape=[1], dtype=tf.int32)
        row_crop_idx = tf.constant(int((IMAGE_SIZE[1] - CROP[2])/2), shape=[1], dtype=tf.int32)
        begin_crop = tf.squeeze(tf.stack([zero_tensor, col_crop_idx, row_crop_idx, zero_tensor]))
        processed_video = tf.slice(clip_op, begin=begin_crop, size=CROP)

#        # reshape to correct size for nework
#        processed_video = tf.reshape(processed_video, [CLIPS_PER_VIDEO, FRAMES_PER_CLIP, CROP[1], CROP[2], CROP[3]])
        
        # normalise single images
        # processed_video = tf.reshape(processed_video, [CLIPS_PER_VIDEO * FRAMES_PER_CLIP, CROP[1], CROP[2], CROP[3]])
        mean, var = tf.nn.moments(processed_video, axes=[0,1,2,3])
        processed_video = tf.nn.batch_normalization(processed_video, mean=mean, variance=var, offset=None, scale=None, variance_epsilon=1e-10)

    return processed_video
    
def read_and_decode(filename_queue):
    readerOptions = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)
    reader = tf.TFRecordReader(options=readerOptions, name='testing_tfreader')
    _, serialized_example = reader.read(filename_queue)

    with tf.name_scope('TFRecordDecoding'):
        _, features_encoded = tf.parse_single_sequence_example(
            serialized_example,
            sequence_features={
                'rgbs':tf.FixedLenSequenceFeature([],dtype=tf.string),
                'sample_id': tf.FixedLenSequenceFeature([], dtype=tf.string),
                'clip_id': tf.FixedLenSequenceFeature([], dtype=tf.string),
                'num_frames': tf.FixedLenSequenceFeature([], dtype=tf.string),
                'padding': tf.FixedLenSequenceFeature([], dtype=tf.string)
            }
        )

        # decode video and apply preprocessing to entire video
        seq_rgb = tf.decode_raw(features_encoded['rgbs'], tf.uint8)
        seq_rgb_pp = video_preprocessing_testing_op(seq_rgb)

        seq_sample_id = tf.decode_raw(features_encoded['sample_id'], tf.int32)
        seq_sample_id = tf.reshape(seq_sample_id, [1])

        seq_clip_id = tf.decode_raw(features_encoded['clip_id'], tf.int32)
        seq_clip_id = tf.reshape(seq_clip_id, [1])

        seq_padding = tf.decode_raw(features_encoded['padding'], tf.int32)
        seq_padding = tf.reshape(seq_padding, [1])
        return seq_rgb_pp, seq_sample_id, seq_clip_id, seq_padding


def input_pipeline(filenames):
    with tf.name_scope('input_pipeline_testing'):

        # Create a input file queue
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=NUM_EPOCHS, 
                                                        shuffle=False, capacity=1000,
                                                        name='Testing_string_input')

        # Read data from .tfrecords files and decode to a list of samples (Using threads)
        rgb, sample_id, clip_id, padding = read_and_decode(filename_queue)

        batch_rgb, batch_sample_id, batch_clip_id, batch_padding = tf.train.batch([rgb, sample_id, clip_id, padding],
                                                        batch_size=BATCH_SIZE,
                                                        capacity=QUEUE_CAPACITY
                                                        )


        #reshape video to correct shape
        batch_rgb = tf.reshape(batch_rgb, [BATCH_SIZE * CLIPS_PER_VIDEO, FRAMES_PER_CLIP, CROP[1], CROP[2], CROP[3]])

        batch_sample_id = tf.reshape(batch_sample_id, [BATCH_SIZE, 1])
        batch_clip_id = tf.reshape(batch_clip_id, [BATCH_SIZE, 1])
        batch_padding = tf.reshape(batch_padding, [BATCH_SIZE, 1])
        

        return batch_rgb, batch_sample_id, batch_clip_id, batch_padding

