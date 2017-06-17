''' Constant Settings  '''
'''#####################'''

# dont use sample 091 since there is a mistake with the labels
TRAIN_ID = range(2,3)         # Raw file format : Sample0001.zip - Sample0470.zip
VALIDATION_ID = range(475,478)  # Raw file format : Sample0471.zip - Sample0700.zip
TEST_ID = range(701,702)        # Raw file format : Sample0701.zip - Sample0941.zip

IMAGE_SIZE = (150, 120, 3)
PADDING_SIZE = (1,) + IMAGE_SIZE

RAW_DATA_PATH = '../data_pp/'
TFRecord_DATA_PATH = '../tf-data/'

# use clips of 80 frames like they did in the paper
FRAMES_PER_CLIP_PP = 14
FRAMES_PER_CLIP = 8 
FRAMES_PER_VIDEO = 8 #80
CLIPS_PER_VIDEO = int(FRAMES_PER_VIDEO / FRAMES_PER_CLIP)

"""preprocesssing parameters"""
CROP = (CLIPS_PER_VIDEO * FRAMES_PER_CLIP, 112, 112, 3) # 1 based crop shape for tf.slice
ROT_ANGLE = 0 #0.083  #+-15deg, it has to be rads
JITTERING = (FRAMES_PER_CLIP,) + IMAGE_SIZE

'''Self-defined gesture labels'''
NO_GESTURE = 21
NUM_OF_NO_GESTURE_CLIPS = 3

'''Training parameters'''
DROPOUT_RATE =0.5 #0.75
LEARNING_RATE = 3e-5 #3e-4
BATCH_SIZE = 30 # 5 gestures per batch
NUM_EPOCHS = 10000
PRINT_EVERY_STEP = 5 #200
EVALUATE_EVERY_STEP = 1000
CHECKPOINT_EVERY_STEP = 500

QUEUE_CAPACITY = int(BATCH_SIZE * 10)


from os import listdir
from os.path import isfile, join

# get paths for all files inside Train, Test and C
train_path = '%s%s' % (TFRecord_DATA_PATH, 'Train_3dcnn')
TRAIN_FILENAMES = [join(train_path, f) for f in listdir(train_path) if isfile(join(train_path, f))]

test_path = '%s%s' % (TFRecord_DATA_PATH, 'Test')
TEST_FILENAMES = sorted([join(test_path, f) for f in listdir(test_path) if isfile(join(test_path, f))])

validation_path = '%s%s' % (TFRecord_DATA_PATH, 'Validation')
VALIDATION_FILENAMES = sorted([join(validation_path, f) for f in listdir(validation_path) if isfile(join(validation_path, f))])

