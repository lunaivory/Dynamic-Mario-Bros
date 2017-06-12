'''Ideas for training '''
# 1. Instead of [0:5][5:10] as frames we can do [0:5][1:6][2:7]...
# 2. Shuffle TRAIN_ID, VALIDATION_ID, etc.

#from training_data_to_tfrecords import get_data_training
from training_data_to_tfrecords_3dcnn import get_data_training
from val_data_to_tfrecords import get_data_val
from testing_data_to_tfrecords_3dcnn import get_data_testing

'''############################'''
'''# Self-defined library     #'''
'''############################'''
from constants import *


if __name__ == '__main__':
    get_data_training(RAW_DATA_PATH, 'Train', TFRecord_DATA_PATH, TRAIN_ID)
    # get_data_val(RAW_DATA_PATH, 'Validation', TFRecord_DATA_PATH, VALIDATION_ID, label_path = RAW_DATA_PATH + 'Validation_reference/')
    get_data_testing(RAW_DATA_PATH, 'Test', TFRecord_DATA_PATH, TEST_ID)
