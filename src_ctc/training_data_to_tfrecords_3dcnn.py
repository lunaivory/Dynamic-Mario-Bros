from ChalearnLAPSample import GestureSample
from tqdm import tqdm
from scipy.misc import imresize
import numpy as np
import tensorflow as tf
from scipy import stats
import util
import time
import math


'''############################'''
'''# Self-defined library     #'''
'''############################'''
from constants import *

'''#####################'''
'''#  Create TFRecord  #'''
'''#####################'''


def get_data_training(path, data_type, write_path, sample_ids):
    class_frequencies = np.zeros(21)
    for sample_id in tqdm(sample_ids):

        '''Get ChaLearn Data reader'''
        sample = GestureSample('%s/%s/Sample%04d.zip' % (path, data_type, sample_id))

        '''Get label per frame'''
        gesture_list = sample.getGestures()
        num_of_frames = sample.getNumFrames()

        labels = []
        ranges = []
        for gesture_id, start_frame, end_frame in gesture_list:
            labels += [gesture_id]
            ranges += [np.arange(start_frame, end_frame, FRAMES_PER_CLIP)[:-1]]

        ranges_lengths = np.asarray([(len(i)-1) for i in ranges])

        # get entire video
        # user = sample.get_entire_user_video()
        vid = sample.get_entire_rgb_video()
        # mask = np.mean(user, axis=3) > 150
        # mask = mask.reshape((mask.shape + (1,)))
        # vid = vid*mask

        videos = []
        dense_label = []
        clip_label = []
        clip_label_video = []

        for lab, id in zip(labels, range(len(labels))):
            for start, id_2 in zip(ranges[id], range(ranges_lengths[id])):


                featureLists = tf.train.FeatureLists(feature_list={
                    'rgbs': util._bytes_feature_list(vid[start:(start+8)]),
                    'label': util._bytes_feature_list(np.asarray((lab - 1,), dtype=np.int32)),
                    'dense_label': util._bytes_feature_list(np.asarray([lab]*8, dtype=np.int32) - 1),
                    'clip_label': util._bytes_feature_list(np.asarray([lab], dtype=np.int32) - 1),
                    'sample_id': util._bytes_feature_list(np.asarray((sample_id,), dtype=np.int32)),
                    'num_frames': util._bytes_feature_list(np.asarray((num_of_frames,), dtype=np.int32))
                })

                sequence_example = tf.train.SequenceExample(feature_lists=featureLists)

                '''Write to .tfrecord file'''

                tf_write_option = tf.python_io.TFRecordOptions(
                    compression_type=tf.python_io.TFRecordCompressionType.GZIP)
                filename = '%s/%s/Sample%04d_%02d_%02d.tfrecords' % (write_path, data_type, sample_id, id, id_2)
                tf_writer = tf.python_io.TFRecordWriter(filename, options=tf_write_option)
                tf_writer.write(sequence_example.SerializeToString())
                tf_writer.close()



