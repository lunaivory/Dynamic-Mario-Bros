'''Ideas for training '''
# 1. Instead of [0:5][5:10] as frames we can do [0:5][1:6][2:7]...
# 2. Shuffle TRAIN_ID, VALIDATION_ID, etc.

# from training_data_to_tfrecords import get_data_training
# from training_data_to_tfrecords_3dcnn_last import get_data_training
#from val_data_to_tfrecords import get_data_val
#from testing_data_to_tfrecords_3dcnn import get_data_testing
import training_data_to_tfrecords
import training_data_to_tfrecords_3dcnn_last

'''############################'''
'''# Self-defined library     #'''
'''############################'''
# from constants import *
# from constants_3dcnn import *
import constants
import constants_3dcnn


if __name__ == '__main__':
    training_data_to_tfrecords_3dcnn_last.get_data_training(constants.RAW_DATA_PATH, 'Train_3dcnn', constants.TFRecord_DATA_PATH, constants_3dcnn.TRAIN_ID)
    training_data_to_tfrecords.get_data_training(constants.RAW_DATA_PATH, 'Train', constants.TFRecord_DATA_PATH, constants.TRAIN_ID)
    #training_data_to_tfrecords_3dcnn.get_data_training(RAW_DATA_PATH, 'Test', TFRecord_DATA_PATH, constants.TEST_ID)
    # get_data_val(RAW_DATA_PATH, 'Validation', TFRecord_DATA_PATH, VALIDATION_ID, label_path = RAW_DATA_PATH + 'Validation_reference/')
    # get_data_testing(RAW_DATA_PATH, 'Test', TFRecord_DATA_PATH, TEST_ID)
