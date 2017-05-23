'''#####################'''
'''# Constant Settings #'''
'''#####################'''

TRAIN_ID = range(3,11)         # Raw file format : Sample0001.zip - Sample0470.zip
VALIDATION_ID = range(471,481)  # Raw file format : Sample0471.zip - Sample0700.zip
TEST_ID = range(701,711)        # Raw file format : Sample0701.zip - Sample0941.zip
# TRAIN_LIST = range(1,471)         # Raw file format : Sample0001.zip - Sample0470.zip
# VALIDATION_LIST = range(471,701)  # Raw file format : Sample0471.zip - Sample0700.zip
# TEST_LIST = range(701,941)        # Raw file format : Sample0701.zip - Sample0941.zip

CROP = (10, 330, 140, 460)
IMAGE_SIZE = (320, 320, 3)
# RESIZE_RATIO = 0.25

RAW_DATA_PATH = '../data/'
TFRecord_DATA_PATH = '../tf-data/'

FRAMES_PER_CLIP = 8 
MAX_FRAMES = 2000
CLIPS_PER_VIDEO = int(MAX_FRAMES / FRAMES_PER_CLIP)

'''Self-defined gesture labels'''
NO_GESTURE = 21
EMPTY_PADDING = 22

'''Training parameters'''
DROPOUT_RATE = 0.75
LEARNING_RATE = 5e-4
BATCH_SIZE = 1 #128
NUM_EPOCHS = 10000
PRINT_EVERY_STEP = 200
EVALUATE_EVERY_STEP = 1000
CHECKPOINT_EVERY_STEP = 1000

NUM_READ_THREADS = 1
QUEUE_CAPACITY = BATCH_SIZE * 2

TRAIN_FILENAMES = ['%s/%s/Sample%04d.tfrecords' % (TFRecord_DATA_PATH, 'Train', i) for i in TRAIN_ID]
TEST_FILENAMES = ['%s/%s/Sample%04d.tfrecords' % (TFRecord_DATA_PATH, 'Test', i) for i in TEST_ID]
VALIDATION_FILENAMES = ['%s/%s/Sample%04d.tfrecords' % (TFRecord_DATA_PATH, 'Validation', i) for i in VALIDATION_ID]
