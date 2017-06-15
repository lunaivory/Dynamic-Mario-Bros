from ChalearnLAPSample import GestureSample
from tqdm import tqdm
from scipy.misc import imresize
import numpy as np
import tensorflow as tf
from scipy import stats
import util
import math
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

    for sample_id in tqdm(sample_ids):

        '''Get ChaLearn Data reader'''
        sample = GestureSample('%s/%s/Sample%04d.zip' % (path, data_type, sample_id))

        '''Get label per frame'''
        gesture_list = sample.getGestures()
        num_of_frames = sample.getNumFrames()

        frames_range = list(np.arange(0, num_of_frames, 400)[:-1])
        frames_range.append(num_of_frames)

        labels = []
        ranges = []

        dense_label = np.zeros(num_of_frames)
        dense_label[:] = NO_GESTURE

        for gesture_id, start_frame, end_frame in gesture_list:
            labels += [gesture_id]
            dense_label[(start_frame-1):end_frame] = gesture_id

        #get also clip labels
        clip_label_range = np.arange(0, num_of_frames, 8)
        clip_labels = []

        for clip_label in clip_label_range:
            clip_dense_labels_slice = dense_label[clip_label: clip_label+8]
            lab_truth = clip_dense_labels_slice != NO_GESTURE
            n = np.sum(lab_truth)
            if n > 5:
                clip_labels.append(clip_dense_labels_slice[lab_truth][0])
            else:
                clip_labels.append(NO_GESTURE)

        clip_labels = np.asarray(clip_labels, dtype=np.int32)
        # get entire video
        user = sample.get_entire_user_video()
        vid = sample.get_entire_rgb_video()
        mask = np.mean(user, axis=3) > 150
        mask = mask.reshape((mask.shape + (1,)))
        vid = vid*mask

        for id in range(len(frames_range[:-1])):
            start = frames_range[id]
            end = frames_range[id+1]
            featureLists = tf.train.FeatureLists(feature_list={
                'rgbs': util._bytes_feature_list(vid[start:end]),
                'label': util._bytes_feature_list(np.asarray((clip_labels[id] - 1,), dtype=np.int32)),
                'dense_label': util._bytes_feature_list(np.asarray(dense_label[start:end], dtype=np.int32) - 1),
                'clip_label': util._bytes_feature_list(np.asarray(clip_labels[math.floor(start/8):math.floor(end/8)], dtype=np.int32) - 1),
                'sample_id': util._bytes_feature_list(np.asarray((sample_id,), dtype=np.int32)),
                'num_frames': util._bytes_feature_list(np.asarray((num_of_frames,), dtype=np.int32))
            })

            sequence_example = tf.train.SequenceExample(feature_lists=featureLists)

            '''Write to .tfrecord file'''

            tf_write_option = tf.python_io.TFRecordOptions(
                compression_type=tf.python_io.TFRecordCompressionType.GZIP)
            filename = '%s/%s/Sample%04d_%02d.tfrecords' % (write_path, data_type, sample_id, id)
            tf_writer = tf.python_io.TFRecordWriter(filename, options=tf_write_option)
            tf_writer.write(sequence_example.SerializeToString())
            tf_writer.close()



