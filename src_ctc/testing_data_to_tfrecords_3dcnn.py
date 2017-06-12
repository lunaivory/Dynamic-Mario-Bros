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


def get_data_testing(path, data_type, write_path, sample_ids):
    class_frequencies = np.zeros(21)
    for sample_id in tqdm(sample_ids):

        '''Get ChaLearn Data reader'''
        sample = GestureSample('%s/%s/Sample%04d.zip' % (path, data_type, sample_id))

        num_of_frames = sample.getNumFrames()

        # get entire video
        vid = sample.get_entire_rgb_video()

        videos = []

        clip_id = 0

        for start in range(0, num_of_frames, FRAMES_PER_CLIP):
            ## pad the clip if it exceed the length of the video
            padding = 0
            if (start + FRAMES_PER_CLIP > num_of_frames):
                padding = FRAMES_PER_CLIP - (num_of_frames - start)
                clip = np.concatenate((vid[start:num_of_frames], np.asarray([vid[0]*(padding)])), axis=0)
            else:
                clip = vid[start:(start+FRAMES_PER_CLIP)]

            featureLists = tf.train.FeatureLists(feature_list={
                'rgbs': util._bytes_feature_list(clip),
                'sample_id': util._bytes_feature_list(np.asarray((sample_id,), dtype=np.int32)),
                'clip_id': util._bytes_feature_list(np.asarray((clip_id,), dtype=np.int32)),
                'num_frames': util._bytes_feature_list(np.asarray((num_of_frames,), dtype=np.int32)),
                'padding': util._bytes_feature_list(np.asarray((padding,), dtype=np.int32))
            })

            sequence_example = tf.train.SequenceExample(feature_lists=featureLists)

            tf_write_option = tf.python_io.TFRecordOptions(
                compression_type=tf.python_io.TFRecordCompressionType.GZIP)

            filename = '%s/%s/Sample%04d_%02d.tfrecords' % (write_path, data_type, sample_id, clip_id)
            clip_id += 1

            tf_writer = tf.python_io.TFRecordWriter(filename, options=tf_write_option)
            tf_writer.write(sequence_example.SerializeToString())
            tf_writer.close()
