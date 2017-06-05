'''Ideas for training '''
# 1. Instead of [0:5][5:10] as frames we can do [0:5][1:6][2:7]...
# 2. Shuffle TRAIN_ID, VALIDATION_ID, etc.

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

def get_padding(video):
    """Get padding for last clip from video"""
    # This is used to pad gestures that are too short
    last_frame=video[-1]

    padding = list(np.zeros((FRAMES_PER_VIDEO,) + np.shape(last_frame)))

    return padding

def get_data_test(path, data_type, write_path, sample_ids):
    for sample_id in tqdm(sample_ids):

        '''Get ChaLearn Data reader'''
        sample = GestureSample('%s/%s/Sample%04d.zip' % (path, data_type, sample_id))

        '''Get label per frame'''
        num_of_frames = sample.getNumFrames()
        num_videos = math.ceil(num_of_frames/FRAMES_PER_VIDEO)

        # get entire video
        vid = sample.get_entire_rgb_video()

        '''Split it into videos of MAX_FRAMES (80 as in the paper) frames'''
        # padding = np.zeros(IMAGE_SIZE, dtype=np.uint8)
        padding = get_padding(vid)
        end_padding = 0
        videos = []

        for video_range in range(num_videos):
            start = video_range*FRAMES_PER_VIDEO
            end = (video_range+1)*FRAMES_PER_VIDEO

            if end > num_of_frames:
                end_padding = end - num_of_frames
                end = num_of_frames

            single_video = [list(vid[start:end]) + padding[:end_padding]]
            single_video = np.asarray(single_video, dtype=np.uint8).reshape((int(FRAMES_PER_VIDEO / FRAMES_PER_CLIP),
                                                                             FRAMES_PER_CLIP) + (IMAGE_SIZE))

            videos += [single_video]

            end_padding = 0

        for gesture_video, ind in zip(videos, range(num_videos)):
            '''Create TFRecord structure'''
            # context = tf.train.Features(feature={'sample_id': util._int64_feature(sample_id)
            #                                     })

            featureLists = tf.train.FeatureLists(feature_list={
                'rgbs': util._bytes_feature_list(gesture_video),
                'label': util._bytes_feature_list(np.asarray((0,), dtype=np.int32)),
                'sample_id': util._bytes_feature_list(np.asarray((sample_id,), dtype=np.int32)),
                'num_frames': util._bytes_feature_list(np.asarray((num_of_frames,), dtype=np.int32)),
            })

            sequence_example = tf.train.SequenceExample(feature_lists=featureLists)

            '''Write to .tfrecord file'''

            tf_write_option = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)
            filename = '%s/%s/Sample%04d_%02d.tfrecords' % (write_path, data_type, sample_id, ind)
            tf_writer = tf.python_io.TFRecordWriter(filename, options=tf_write_option)
            tf_writer.write(sequence_example.SerializeToString())
            tf_writer.close()