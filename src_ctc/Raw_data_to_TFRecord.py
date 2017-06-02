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
    time_intervals = np.asarray(gestures)[:,1:].ravel()
    range_1 = np.arange(1, time_intervals.shape[0]-1,step=2)
    range_2 = np.arange(2, time_intervals.shape[0],step=2)
    time_diff = time_intervals[range_2] - time_intervals[range_1]
    max_time_ind = np.argmax(time_diff)

    idx_2 = range_2[max_time_ind]
    idx_1 = range_1[max_time_ind]

    start_frame = time_intervals[idx_1]
    end_frame = time_intervals[idx_2]

    padding = list(video[start_frame:end_frame-22])

    # make sure padding  has enough frame to not run in out of range problems later
    while len(padding) < int(FRAMES_PER_VIDEO/2):
        padding *= 2

    padding = padding[:int(FRAMES_PER_VIDEO/2)]

    return padding



def get_data(path, data_type, write_path, sample_ids, label_path = None):
    for sample_id in tqdm(sample_ids):
        
        '''Get ChaLearn Data reader'''
        if label_path is not None: #data_type == 'Validation'
            sample = GestureSample('%s/%s/Sample%04d.zip'%(path, data_type, sample_id), 
                                   labelFileName=label_path + 'Sample%04d_prediction.csv'%sample_id)
        else: # data_type == 'Test', 'Train'
            sample = GestureSample('%s/%s/Sample%04d.zip'%(path, data_type, sample_id))
            
        '''Get label per frame'''
        gesture_list = sample.getGestures()
        num_of_frames = sample.getNumFrames()

        labels = []
        mid_frame = []
        for gesture_id, start_frame, end_frame in gesture_list:
            labels += [gesture_id]
            mid_frame += [round((start_frame+end_frame)/2)]
        labels = labels + [0] * (FRAMES_PER_CLIP - len(labels))

        # get entire video
        vid = sample.get_entire_rgb_video()
            
        '''Split it into videos of MAX_FRAMES (80 as in the paper) frames'''
        # padding = np.zeros(IMAGE_SIZE, dtype=np.uint8)
        padding = get_padding(vid, gesture_list)
        start_padding = 0
        end_padding = 0
        videos = []

        for f, id in zip(mid_frame, range(len(labels))):
            start = f-40
            end = f+40

            if start < 0:
                start_padding = -start
                start = 0

            if end > num_of_frames:
                end_padding = end - num_of_frames
                end = num_of_frames


            if (start < gesture_list[id-1][2]) and (id > 0):
                start_padding = gesture_list[id-1][2] - start
                start = gesture_list[id-1][2]

            if id < (len(labels) - 1):
                if (end > gesture_list[id+1][1]):
                    end_padding = end - gesture_list[id+1][1]
                    end = gesture_list[id+1][1]

            single_video = [padding[:start_padding] + list(vid[start:end]) + padding[:end_padding]]
            single_video = np.asarray(single_video, dtype=np.uint8).reshape((int(FRAMES_PER_VIDEO/FRAMES_PER_CLIP),
                                                                             FRAMES_PER_CLIP)+(IMAGE_SIZE))

            videos += [single_video]
            start_padding = 0
            end_padding = 0

        for gesture_video, label, ind in zip(videos, labels, range(len(labels))):

            '''Create TFRecord structure'''
            context = tf.train.Features(feature={'sample_id': util._int64_feature(sample_id),
                                                 'gesture_list_len': util._int64_feature(1)})

            featureLists = tf.train.FeatureLists(feature_list={
                'rgbs':util._bytes_feature_list(gesture_video),
                'label':util._bytes_feature_list(np.asarray((label,), dtype=np.int32))
            })

            sequence_example = tf.train.SequenceExample(context=context, feature_lists=featureLists)

            '''Write to .tfrecord file'''

            tf_write_option = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)
            filename = '%s/%s/Sample%04d_%02d.tfrecords' % (write_path, data_type, sample_id, ind)
            tf_writer = tf.python_io.TFRecordWriter(filename, options=tf_write_option)
            tf_writer.write(sequence_example.SerializeToString())
            tf_writer.close()



if __name__ == '__main__':
    get_data(RAW_DATA_PATH, 'Train', TFRecord_DATA_PATH, TRAIN_ID)
    get_data(RAW_DATA_PATH, 'Validation', TFRecord_DATA_PATH, VALIDATION_ID, label_path = RAW_DATA_PATH + 'Validation_reference/')
    #get_data(RAW_DATA_PATH, 'Test', TFRecord_DATA_PATH, TEST_ID)
