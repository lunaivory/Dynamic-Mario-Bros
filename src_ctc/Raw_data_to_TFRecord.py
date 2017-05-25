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
        if (num_of_frames > MAX_FRAMES):
            raise Exception('Sample %d has %d Frames (> MAX_FRAMES:%d)' %(sample_id, num_of_frames, MAX_FRAMES))
            
        label = [EMPTY_PADDING] + [NO_GESTURE]* num_of_frames + [EMPTY_PADDING] * (MAX_FRAMES - num_of_frames)
        has_label = True if len(gesture_list) > 0 else False
        for gesture_id, start_frame, end_frame in gesture_list:
            label[start_frame:end_frame+1] = [gesture_id]*(end_frame + 1 - start_frame)
        
        '''Get sliced and cropped RGB data from the whole sample'''
        rgb = [np.zeros(IMAGE_SIZE)]
        for f in range(1, MAX_FRAMES+1):
            if (f > num_of_frames): # Add 0 paddings
                rgb_data = np.zeros(IMAGE_SIZE, dtype=np.uint8)
            else: 
                segmentation_mask = sample.getUser(f)
                if (segmentation_mask.sum() == 0):
                    label[f] = EMPTY_PADDING
                    print('Empty segmentation mask for Sample %d on frame %d' % (sample_id, f))
                rgb_data = sample.getRGB(f) * segmentation_mask
                rgb_data = rgb_data[CROP[0]:CROP[1], CROP[2]:CROP[3],:]
            rgb += [rgb_data]
            
        '''Make it into clips'''
        rgbs = []
        labels = []
        for f in range(1, MAX_FRAMES+1, FRAMES_PER_CLIP):
            rgbs += [np.asarray(rgb[f:f+FRAMES_PER_CLIP])]
            labels += [int(stats.mode(label[f:f+FRAMES_PER_CLIP])[0])]
            
        '''Create TFRecord structure'''
        context = tf.train.Features(feature={'sample_id': util._int64_feature(sample_id)})

        # Create sparse tensor for CTC lost
        indices, values, shapes = util.sparse_tuple_from([labels])
        labels = tf.SparseTensor(indices, values, shapes)
            
        featureLists = tf.train.FeatureLists(feature_list={
            'rgbs':util._bytes_feature_list(rgbs),
            'labels_index': util._bytes_feature_list(indices),
            'labels_value': util._bytes_feature_list(values),
            'labels_shape': util._bytes_feature_list(shapes)
        })
       
        sequence_example = tf.train.SequenceExample(context=context, feature_lists=featureLists)
        
        '''Write to .tfrecord file'''
    
        tf_write_option = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)
        filename = '%s/%s/Sample%04d.tfrecords' % (write_path, data_type, sample_id)
        tf_writer = tf.python_io.TFRecordWriter(filename, options=tf_write_option)
        tf_writer.write(sequence_example.SerializeToString())
        tf_writer.close()



if __name__ == '__main__':
    get_data(RAW_DATA_PATH, 'Train', TFRecord_DATA_PATH, TRAIN_ID)
    #get_data(RAW_DATA_PATH, 'Validation', TFRecord_DATA_PATH, VALIDATION_ID, label_path = RAW_DATA_PATH + 'Validation_reference/')
    #get_data(RAW_DATA_PATH, 'Test', TFRecord_DATA_PATH, TEST_ID)
