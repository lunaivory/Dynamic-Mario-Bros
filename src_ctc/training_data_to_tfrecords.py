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


def get_padding(video, gestures):
    """Get no_gestrue padding from video"""
    # This is used to pad gestures that are too short
    time_intervals = np.asarray(gestures)[:, 1:].ravel()
    range_1 = np.arange(1, time_intervals.shape[0] - 1, step=2)
    range_2 = np.arange(2, time_intervals.shape[0], step=2)
    time_diff = time_intervals[range_2] - time_intervals[range_1]
    max_time_ind = np.argmax(time_diff)

    idx_2 = range_2[max_time_ind]
    idx_1 = range_1[max_time_ind]

    start_frame = time_intervals[idx_1]
    end_frame = time_intervals[idx_2]

    padding = list(video[start_frame:end_frame])

    # make sure padding  has enough frame to not run in out of range problems later
    while len(padding) < int(FRAMES_PER_VIDEO / 2):
        padding *= 2

    padding = padding[:int(FRAMES_PER_VIDEO)]

    return padding


def get_data_training(path, data_type, write_path, sample_ids):
    for sample_id in tqdm(sample_ids):

        '''Get ChaLearn Data reader'''
        sample = GestureSample('%s/%s/Sample%04d.zip' % (path, data_type, sample_id))

        '''Get label per frame'''
        gesture_list = sample.getGestures()
        num_of_frames = sample.getNumFrames()

        labels = []
        mid_frame = []
        for gesture_id, start_frame, end_frame in gesture_list:
            labels += [gesture_id]
            mid_frame += [round((start_frame + end_frame) / 2)]

        # get entire video
        vid = sample.get_entire_rgb_video()
        user = sample.get_entire_user_video()
        mask = np.mean(user, axis=3) > 150
        mask = mask.reshape((mask.shape + (1,)))
        vid = vid * mask

        '''Split it into videos of MAX_FRAMES (80 as in the paper) frames'''
        # padding = np.zeros(IMAGE_SIZE, dtype=np.uint8)
        padding = get_padding(vid, gesture_list)
        start_padding = 0
        end_padding = 0
        videos = []
        dense_label = []
        clip_label = []
        clip_label_video = []

        for f, lab, id in zip(mid_frame, labels, range(len(labels))):
            start = f - 40
            end = f + 40

            label_padding_start = abs(start - gesture_list[id][1])
            label_padding_end = abs(gesture_list[id][2] - end)
            label_gesture = gesture_list[id][2] - gesture_list[id][1]

            if start < 0:
                start_padding = -start
                start = 0

            if end > num_of_frames:
                end_padding = end - num_of_frames
                end = num_of_frames

            if (start < gesture_list[id - 1][2]) and (id > 0):
                start_padding = gesture_list[id - 1][2] - start
                start = gesture_list[id - 1][2]

            if id < (len(labels) - 1):
                if (end > gesture_list[id + 1][1]):
                    end_padding = end - gesture_list[id + 1][1]
                    end = gesture_list[id + 1][1]

            single_video = [padding[:start_padding] + list(vid[start:end]) + padding[:end_padding]]
            single_video = np.asarray(single_video, dtype=np.uint8).reshape((int(FRAMES_PER_VIDEO / FRAMES_PER_CLIP),
                                                                             FRAMES_PER_CLIP) + (IMAGE_SIZE))

            # get frame by frame labels to calculate accuracy during training and Jaccard score for val/test
            dense_lab = label_padding_start * [NO_GESTURE] + label_gesture * [lab] + label_padding_end * [NO_GESTURE]
            for i in range(0,FRAMES_PER_VIDEO, FRAMES_PER_CLIP):
                extracted_labels = np.asarray(dense_lab[i: i+FRAMES_PER_CLIP]) == lab
                if np.sum(extracted_labels) < 4:
                    clip_label_video += [NO_GESTURE]
                else:
                    clip_label_video += [lab]

            videos += [single_video]
            dense_label += [dense_lab]
            clip_label += [clip_label_video]
            start_padding = 0
            end_padding = 0
            clip_label_video = []

        #add also padding video
        videos += [np.asarray(padding, dtype=np.uint8)]
        dense_label += [[NO_GESTURE]*FRAMES_PER_VIDEO]
        clip_label += [[NO_GESTURE]*int(FRAMES_PER_VIDEO/FRAMES_PER_CLIP)]

        for gesture_video, label, ind in zip(videos, labels, range(len(labels))):
            '''Create TFRecord structure'''
            # context = tf.train.Features(feature={'sample_id': util._int64_feature(sample_id),
            #                                     })

            featureLists = tf.train.FeatureLists(feature_list={
                'rgbs': util._bytes_feature_list(gesture_video),
                'label': util._bytes_feature_list(np.asarray((label-1,), dtype=np.int32)),
                'dense_label': util._bytes_feature_list(np.asarray(dense_label[ind], dtype=np.int32)-1),
                'clip_label': util._bytes_feature_list(np.asarray(clip_label[ind], dtype=np.int32) - 1),
                'sample_id': util._bytes_feature_list(np.asarray((sample_id,), dtype=np.int32)),
                'num_frames': util._bytes_feature_list(np.asarray((num_of_frames,), dtype=np.int32))
            })

            sequence_example = tf.train.SequenceExample(feature_lists=featureLists)

            '''Write to .tfrecord file'''

            tf_write_option = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)
            filename = '%s/%s/Sample%04d_%02d.tfrecords' % (write_path, data_type, sample_id, ind)
            tf_writer = tf.python_io.TFRecordWriter(filename, options=tf_write_option)
            tf_writer.write(sequence_example.SerializeToString())
            tf_writer.close()


