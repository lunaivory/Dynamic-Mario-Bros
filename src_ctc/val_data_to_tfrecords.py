from ChalearnLAPSample import GestureSample
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import util
import math


'''############################'''
'''# Self-defined library     #'''
'''############################'''
from constants import *

'''#####################'''
'''#  Create TFRecord  #'''
'''#####################'''

def get_padding(video):
    """Get padding for last clip from video"""
    # This is used to pad gestures that are too short
    last_frame=video[-1]

    padding = list(np.zeros((FRAMES_PER_VIDEO,) + np.shape(last_frame)))

    return padding

def get_data_val(path, data_type, write_path, sample_ids, label_path):
    for sample_id in tqdm(sample_ids):

        if label_path is not None:
            '''Get ChaLearn Data reader'''
            sample = GestureSample('%s/%s/Sample%04d.zip' % (path, data_type, sample_id),
                                   labelFileName=label_path + 'Sample%04d_prediction.csv' % sample_id)

        '''Get label per frame'''
        gesture_list = sample.getGestures()
        num_of_frames = sample.getNumFrames()
        num_videos = math.ceil(num_of_frames/FRAMES_PER_VIDEO)

        labels = []
        mid_frame = []
        for gesture_id, start_frame, end_frame in gesture_list:
            labels += [gesture_id]
            mid_frame += [round((start_frame + end_frame) / 2)]
        labels = labels + [0] * (FRAMES_PER_CLIP - len(labels))

        # get entire video
        vid = sample.get_entire_rgb_video()

        '''Split it into videos of MAX_FRAMES (80 as in the paper) frames'''
        padding = get_padding(vid)
        end_padding = 0
        videos = []

        dense_label = np.zeros((num_of_frames), dtype=np.int64)
        dense_label[:] = NO_GESTURE
        dense_label_list = []

        gesture_array = np.asarray(gesture_list)[:, 1:]
        for start, end, id in zip(gesture_array[:,0], gesture_array[:,1], range(len(labels))):
            dense_label[start:end] = labels[id]

        for video_range in range(num_videos):
            start = video_range*FRAMES_PER_VIDEO
            end = (video_range+1)*FRAMES_PER_VIDEO

            if end > num_of_frames:
                end_padding = end - num_of_frames
                end = num_of_frames

            single_video = [list(vid[start:end]) + padding[:end_padding]]
            single_video = np.asarray(single_video, dtype=np.uint8).reshape((int(FRAMES_PER_VIDEO / FRAMES_PER_CLIP),
                                                                             FRAMES_PER_CLIP) + (IMAGE_SIZE))

            dense_lab = [list(dense_label[start:end]) + end_padding * [NO_GESTURE]]

            videos += [single_video]
            dense_label_list += [dense_lab]

            end_padding = 0

        for gesture_video, label, ind in zip(videos, dense_label_list, range(len(dense_label_list))):
            '''Create TFRecord structure'''
            # context = tf.train.Features(feature={'sample_id': util._int64_feature(sample_id)
            #                                     })

            featureLists = tf.train.FeatureLists(feature_list={
                'rgbs': util._bytes_feature_list(gesture_video),
                'label': util._bytes_feature_list(np.asarray(label, dtype=np.int32)-1),
                'sample_id': util._bytes_feature_list(np.asarray((sample_id,), dtype=np.int32)),
                'num_frames': util._bytes_feature_list(np.asarray((num_of_frames,), dtype=np.int32)),
                'ind': util._bytes_feature_list(np.asarray((ind,), dtype=np.int32))
            })

            sequence_example = tf.train.SequenceExample(feature_lists=featureLists)

            '''Write to .tfrecord file'''

            tf_write_option = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)
            filename = '%s/%s/Sample%04d_%02d.tfrecords' % (write_path, data_type, sample_id, ind)
            tf_writer = tf.python_io.TFRecordWriter(filename, options=tf_write_option)
            tf_writer.write(sequence_example.SerializeToString())
            tf_writer.close()