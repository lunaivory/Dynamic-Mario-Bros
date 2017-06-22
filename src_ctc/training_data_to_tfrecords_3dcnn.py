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

gesture_cnt = [0, 0]

def get_data_training(path, data_type, write_path, sample_ids):

    isTrain = data_type.find('Train') >= 0

    for sample_id in tqdm(sample_ids):

        '''Get ChaLearn Data reader'''
        sample = GestureSample('%s/%s/Sample%04d.zip' % (path, data_type, sample_id))

        '''Get label per frame'''
        gesture_list = sample.getGestures()
            
        num_of_frames = sample.getNumFrames()

        # get entire video
        user = sample.get_entire_user_video()
        vid = sample.get_entire_rgb_video()
        mask = np.mean(user, axis=3) > 150
        mask = mask.reshape((mask.shape + (1,)))
        vid = vid*mask

        labels = []

        #get also clip labels
        clip_label_range = np.arange(0, num_of_frames, FRAMES_PER_CLIP)
        dense_label = np.zeros(num_of_frames)
        dense_label[:] = NO_GESTURE
        clip_labels = []

        cut_dense_labels = []
        cut_clip_labels = []
        cut_vid = np.zeros([8, 150, 120, 3], np.uint8)

        for gesture_id, start_frame, end_frame in gesture_list:
            labels += [gesture_id]
            dense_label[(start_frame-1):end_frame] = gesture_id

        for clip_label in clip_label_range:
            clip_dense_labels_slice = dense_label[clip_label: clip_label+FRAMES_PER_CLIP]
            lab_truth = clip_dense_labels_slice != NO_GESTURE
            n = np.sum(lab_truth)
            if n > 5:
                cut_clip_labels.append(clip_dense_labels_slice[lab_truth][0])
                #cut_vid+= [vid[clip_label : clip_label + FRAMES_PER_CLIP]]
                cut_vid = np.concatenate((cut_vid, vid[clip_label : clip_label + FRAMES_PER_CLIP]), axis=0)
                cut_dense_labels += [clip_dense_labels_slice[lab_truth][0]]*FRAMES_PER_CLIP
                gesture_cnt[0] += 1
            elif gesture_cnt[0] > 20 * gesture_cnt[1]:
                cut_clip_labels.append(NO_GESTURE)
                cut_vid = np.concatenate((cut_vid, vid[clip_label : clip_label + FRAMES_PER_CLIP]), axis=0)
                #cut_vid += [vid[clip_label : clip_label + FRAMES_PER_CLIP]]
                cut_dense_labels += [NO_GESTURE] * FRAMES_PER_CLIP
                gesture_cnt[1] += 1

        cut_clip_labels = np.asarray(cut_clip_labels, dtype=np.int32)
        cut_vid = cut_vid[8:]
        #cut_vid = np.asarray(cut_vid, dtype = np.uint8)
        #cut_vid = np.reshape(cut_vid, (cut_vid.shape[0] * cut_vid.shape[1], cut_vid.shape[2], cut_vid.shape[3], cut_vid.shape[4]))
        cut_dense_labels = np.asarray(cut_dense_labels, dtype=np.int32)


        num_of_frames = cut_dense_labels.shape[0]

        frames_range = list(np.arange(0, num_of_frames, FRAMES_PER_VIDEO_PP)[:-1])
        frames_range.append(num_of_frames)

        for id in range(len(frames_range[:-1])):
            start = frames_range[id]
            end = frames_range[id+1]

            clip_label_slice = np.asarray(cut_clip_labels[math.floor(start/8):math.floor(end/8)], dtype=np.int32)
            featureLists = tf.train.FeatureLists(feature_list={
                'rgbs': util._bytes_feature_list(cut_vid[start:end]),
                'label': util._bytes_feature_list(np.asarray((cut_clip_labels[id] - 1,), dtype=np.int32)),
                'dense_label': util._bytes_feature_list(np.asarray(cut_dense_labels[start:end], dtype=np.int32) - 1),
                'clip_label': util._bytes_feature_list(clip_label_slice- 1),
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

        print(gesture_cnt)



