from ChalearnLAPSample import GestureSample
from tqdm import tqdm
from scipy.misc import imresize
import numpy as np
import tensorflow as tf
from scipy import stats
import util
import time
import math
import random


'''############################'''
'''# Self-defined library     #'''
'''############################'''
import constants_3dcnn

'''#####################'''
'''#  Create TFRecord  #'''
'''#####################'''

def get_no_gesture(gestures):
    """Get no_gestrue padding from video"""
    # This is used to pad gestures that are too short
    time_intervals = np.asarray(gestures)[:, 1:].ravel()
    range_1 = np.arange(1, time_intervals.shape[0] - 1, step=2)
    range_2 = np.arange(2, time_intervals.shape[0], step=2)
    time_diff = time_intervals[range_2] - time_intervals[range_1]

    no_gesture = []

    for i in range(constants_3dcnn.NUM_OF_NO_GESTURE_CLIPS):

        selected_interval = np.random.randint(low=0, high=time_diff.shape[0])

        # sample an interval of no-gestures until you get one larger than 8 clips
        # add extra frames to make sure we get a "true" no gesture and not a noisy one right
        # after a gesture was completed
        extra_frames = np.random.randint(low=0, high=16)

        while time_diff[selected_interval] < (constants_3dcnn.FRAMES_PER_CLIP_PP + extra_frames) :
            selected_interval = np.random.randint(low=0, high=time_diff.shape[0])

        time_ind = selected_interval

        idx_1 = range_1[time_ind]

        start_frame = time_intervals[idx_1]

        no_gesture.append(start_frame + extra_frames)

    return np.asarray(no_gesture)

def get_data_training(path, data_type, write_path, sample_ids):

    is_gesture = 0
    no_gesture = 0
    count = [0] * 21
    quota = {}
    for i in range(0, 21):
        quota[i] = []

    for sample_id in tqdm(sample_ids):

        '''Get ChaLearn Data reader'''
        sample = GestureSample('%s/%s/Sample%04d.zip' % (path, data_type, sample_id))

        '''Get label per frame'''
        gesture_list = sample.getGestures()
        num_of_frames = sample.getNumFrames()

        dense_label = np.zeros(num_of_frames)
        dense_label[:] = constants_3dcnn.NO_GESTURE

        for gesture_id, start_frame, end_frame in gesture_list:
            dense_label[start_frame:end_frame] = gesture_id

        range_num = np.arange(0, num_of_frames, constants_3dcnn.FRAMES_PER_CLIP_PP)[:-1]

        # no_gesture_ranges = get_no_gesture(gesture_list)
        # ranges_lengths.append(NUM_OF_NO_GESTURE_CLIPS)
        # ranges.append(no_gesture_ranges)
        # labels.append(NO_GESTURE)

        # get entire video
        user = sample.get_entire_user_video()
        vid = sample.get_entire_rgb_video()
        mask = np.mean(user, axis=3) > 150
        mask = mask.reshape((mask.shape + (1,)))
        vid = vid * mask

        id = 0
        for rang in range_num:
            counter = np.zeros(shape=22)
            clip = vid[rang:(rang+constants_3dcnn.FRAMES_PER_CLIP_PP)]
            clip_dense_label = dense_label[rang:(rang+constants_3dcnn.FRAMES_PER_CLIP_PP)]
            check = np.sum(clip_dense_label != constants_3dcnn.NO_GESTURE)

            # if most of the frames belong to a gesture label then store the clip under than label
            if check > int(constants_3dcnn.FRAMES_PER_CLIP_PP/2):
                is_gesture += 1

                lab_unique = np.unique(clip_dense_label[clip_dense_label != constants_3dcnn.NO_GESTURE])
                for l in list(lab_unique):
                    counter[int(l)] += 1

                lab = np.argmax(counter)

                featureLists = tf.train.FeatureLists(feature_list={
                    'rgbs': util._bytes_feature_list(clip),
                    'label': util._bytes_feature_list(np.asarray((lab - 1,), dtype=np.int32)),
                    'dense_label': util._bytes_feature_list(np.asarray(clip_dense_label, dtype=np.int32) - 1),
                    'clip_label': util._bytes_feature_list(np.asarray([lab], dtype=np.int32) - 1),
                    'sample_id': util._bytes_feature_list(np.asarray((sample_id,), dtype=np.int32)),
                    'num_frames': util._bytes_feature_list(np.asarray((num_of_frames,), dtype=np.int32))
                })

                count[lab-1] += 1
                if (len(quota[lab-1]) < 100):
                    quota[lab-1].append(featureLists)

                sequence_example = tf.train.SequenceExample(feature_lists=featureLists)

                '''Write to .tfrecord file'''

                tf_write_option = tf.python_io.TFRecordOptions(
                    compression_type=tf.python_io.TFRecordCompressionType.GZIP)
                filename = '%s/%s/Sample%04d_%02d.tfrecords' % (write_path, data_type, sample_id, id)
                tf_writer = tf.python_io.TFRecordWriter(filename, options=tf_write_option)
                tf_writer.write(sequence_example.SerializeToString())
                tf_writer.close()
                id+=1

            #get with prob 30% also some noisy no-gesture frames. Probability is 30% to avoid class imbalance
            elif (check != 0) and no_gesture * 20 < is_gesture:
                lab = constants_3dcnn.NO_GESTURE
                no_gesture += 1

                featureLists = tf.train.FeatureLists(feature_list={
                    'rgbs': util._bytes_feature_list(clip),
                    'label': util._bytes_feature_list(np.asarray((lab - 1,), dtype=np.int32)),
                    'dense_label': util._bytes_feature_list(np.asarray(clip_dense_label, dtype=np.int32) - 1),
                    'clip_label': util._bytes_feature_list(np.asarray([lab], dtype=np.int32) - 1),
                    'sample_id': util._bytes_feature_list(np.asarray((sample_id,), dtype=np.int32)),
                    'num_frames': util._bytes_feature_list(np.asarray((num_of_frames,), dtype=np.int32))
                })

                count[lab-1] += 1
                if (len(quota[lab-1]) < 100):
                    quota[lab-1].append(featureLists)

                sequence_example = tf.train.SequenceExample(feature_lists=featureLists)

                '''Write to .tfrecord file'''

                tf_write_option = tf.python_io.TFRecordOptions(
                    compression_type=tf.python_io.TFRecordCompressionType.GZIP)
                filename = '%s/%s/Sample%04d_%02d.tfrecords' % (write_path, data_type, sample_id, id)
                tf_writer = tf.python_io.TFRecordWriter(filename, options=tf_write_option)
                tf_writer.write(sequence_example.SerializeToString())
                tf_writer.close()
                id += 1

        if (sample_id == sample_ids[-1]):
            max_count = max(count)
            for i in range(21):
                while (count[i] < max_count):
                    count[i] += 1
                    idx = random.randrange(len(quota[i]))
                    sequence_example = tf.train.SequenceExample(feature_lists = quota[i][idx])

                    tf_write_option = tf.python_io.TFRecordOptions(
                        compression_type=tf.python_io.TFRecordCompressionType.GZIP)
                    filename = '%s/%s/Sample%04d_%02d.tfrecords' % (write_path, data_type, sample_id, id)
                    tf_writer = tf.python_io.TFRecordWriter(filename, options=tf_write_option)
                    tf_writer.write(sequence_example.SerializeToString())
                    tf_writer.close()
                    id += 1


    print(is_gesture, no_gesture)
    print(max_count)
    print(count)

#        #also put some non-noisy no gesture
#        no_gesture_ranges = get_no_gesture(gesture_list)
#        for rang in no_gesture_ranges:
#            if (np.random.uniform() < 0.5):
#                clip = vid[rang:(rang + constants_3dcnn.FRAMES_PER_CLIP_PP)]
#                lab = constants_3dcnn.NO_GESTURE
#                clip_dense_label = dense_label[rang:(rang+constants_3dcnn.FRAMES_PER_CLIP_PP)]
#
#                featureLists = tf.train.FeatureLists(feature_list={
#                    'rgbs': util._bytes_feature_list(clip),
#                    'label': util._bytes_feature_list(np.asarray((lab - 1,), dtype=np.int32)),
#                    'dense_label': util._bytes_feature_list(np.asarray(clip_dense_label, dtype=np.int32) - 1),
#                    'clip_label': util._bytes_feature_list(np.asarray([lab], dtype=np.int32) - 1),
#                    'sample_id': util._bytes_feature_list(np.asarray((sample_id,), dtype=np.int32)),
#                    'num_frames': util._bytes_feature_list(np.asarray((num_of_frames,), dtype=np.int32))
#                })
#
#                sequence_example = tf.train.SequenceExample(feature_lists=featureLists)
#
#                '''Write to .tfrecord file'''
#
#                tf_write_option = tf.python_io.TFRecordOptions(
#                compression_type=tf.python_io.TFRecordCompressionType.GZIP)
#                filename = '%s/%s/Sample%04d_%02d.tfrecords' % (write_path, data_type, sample_id, id)
#                tf_writer = tf.python_io.TFRecordWriter(filename, options=tf_write_option)
#                tf_writer.write(sequence_example.SerializeToString())
#                tf_writer.close()
#                id += 1
#


