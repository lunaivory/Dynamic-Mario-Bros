import time
import numpy as np
import tensorflow as tf
from constants import *


# def _int64_feature(value):
#     """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#
#
# def _float_feature(value):
#     """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
#     return tf.train.Feature(int64_list=tf.train.FloatList(value=[value]))
#
#
# def _bytes_feature(value):
#     """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))
#
#
# def _int64_feature_list(values):
#     """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
#     return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])
#
#
# def _bytes_feature_list(values):
#     """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
#     return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])
#
#
# def _float_feature_list(values):
#     """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
#     return tf.train.FeatureList(feature=[_float_feature(v) for v in values])


'''######################################################'''
'''#                   Input Pipeline                   #'''
'''######################################################'''

def video_preprocessing_validation_op(video_op):
    with tf.name_scope('Single_gesture_video_preprocessing_validation'):
        # Reshape for easier preprocessing
        video_op = tf.cast(video_op, dtype=tf.float32)
        clip_op = tf.reshape(video_op, [CLIPS_PER_VIDEO * FRAMES_PER_CLIP] + list(IMAGE_SIZE))

        #### Take central crop of dimension CROP=(CLIPS_PER_VIDEO * FRAMES_PER_CLIP, 112, 112, 3)
        zero_tensor = tf.zeros(shape=[1], dtype=tf.int32)
        col_crop_idx = tf.constant(int((IMAGE_SIZE[0] - CROP[1]) / 2), shape=[1], dtype=tf.int32)
        row_crop_idx = tf.constant(int((IMAGE_SIZE[1] - CROP[2]) / 2), shape=[1], dtype=tf.int32)
        begin_crop = tf.squeeze(tf.stack([zero_tensor, col_crop_idx, row_crop_idx, zero_tensor]))
        processed_video = tf.slice(clip_op, begin=begin_crop, size=CROP)

        # reshape to correct size for nework
        processed_video = tf.reshape(processed_video, [CLIPS_PER_VIDEO, FRAMES_PER_CLIP, CROP[1], CROP[2], CROP[3]])

    return processed_video

def read_and_decode(filename_queue):
    readerOptions = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)
    reader = tf.TFRecordReader(options=readerOptions, name='Validation_tfreader')
    _, serialized_example = reader.read(filename_queue)

    with tf.name_scope('TFRecordDecoding'):
        _, features_encoded = tf.parse_single_sequence_example(
            serialized_example,
            sequence_features={
                'rgbs': tf.FixedLenSequenceFeature([], dtype=tf.string),
                'label': tf.FixedLenSequenceFeature([], dtype=tf.string),
                'sample_id': tf.FixedLenSequenceFeature([], dtype=tf.string),
                'num_frames': tf.FixedLenSequenceFeature([], dtype=tf.string),
                'ind': tf.FixedLenSequenceFeature([], dtype=tf.string)
            }
        )


        # decode video and apply preprocessing to entire video
        seq_rgb = tf.decode_raw(features_encoded['rgbs'], tf.uint8)
        # seq_rgb_pp = video_preprocessing_training_op(seq_rgb)
        seq_rgb_pp = video_preprocessing_validation_op(seq_rgb)

        # decode label and reshape it correct size for shuffle_batch
        seq_label = tf.decode_raw(features_encoded['label'], tf.int32)
        # seq_label = tf.reshape(seq_label, [1])
        seq_label = tf.reshape(seq_label, [FRAMES_PER_VIDEO])

        # decode sample id
        sample_id = tf.decode_raw(features_encoded['sample_id'], tf.int32)
        sample_id = tf.reshape(sample_id, [1])

        # decode number of frames
        num_frames = tf.decode_raw(features_encoded['num_frames'], tf.int32)
        num_frames = tf.reshape(num_frames, [1])

        ind = tf.decode_raw(features_encoded['ind'], tf.int32)
        ind = tf.reshape(ind, [1])

        return seq_rgb_pp, seq_label, sample_id, num_frames, ind, reader


def input_pipeline(filenames):
    with tf.name_scope('input_pipeline_validation'):
        # shuffle input only for training

        # Create a input file queue
        filename_queue = tf.train.string_input_producer(filenames, shuffle=False,
                                                        capacity=1000, name='Validation_string_inputs')

        # Read data from .tfrecords files and decode to a list of samples (Using threads)
        data, label, sample_id, num_frames, ind = read_and_decode(filename_queue)

        # Create batches
        batch_rgb, batch_label, batch_id, batch_num_frames, ind = tf.train.batch(
            [data, label, sample_id, num_frames, ind],
            batch_size=BATCH_SIZE,
            capacity=QUEUE_CAPACITY
            )

        # batch_id and num_frames are only used for validation and testing
        # reshape batch ids tensor
        batch_id = tf.reshape(batch_id, [BATCH_SIZE])

        # reshape num_of frames tensor
        batch_num_frames = tf.reshape(batch_num_frames, [BATCH_SIZE])

        # reshape video to correct shape
        batch_rgb = tf.reshape(batch_rgb, [BATCH_SIZE * CLIPS_PER_VIDEO, FRAMES_PER_CLIP, CROP[1], CROP[2], CROP[3]])

        # keep dense labels to calculate Jaccard similarity
        # values are zero for testing
        batch_label_dense = tf.reshape(batch_label, [BATCH_SIZE*FRAMES_PER_VIDEO])

        ind = tf.reshape(ind, [BATCH_SIZE])

        return batch_rgb, batch_label_dense, batch_id, batch_num_frames,ind



        #
        # if (data_type == 'Train' or data_type == 'Validation'):
        #     return batch_rgb, batch_label_sparse
        # else:
        #     return batch_rgb
