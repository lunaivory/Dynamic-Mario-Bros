import tensorflow as tf
import numpy as np
#import model
import matplotlib.pyplot as plt
import tensorflow.contrib.image
import time as time
import os
from tqdm import tqdm
from ChalearnLAPSample import GestureSample
import math

import util_testing

'''####################################################'''
'''#                Self-defined library              #'''
'''####################################################'''
from constants import *
from sklearn.metrics import jaccard_similarity_score

def exportPredictions(output_csv_file_path, seq_gestures, seq_paddings):
    """ Export the given prediction to the csv file in the given path """
    ''' Given a list of gestures of clips '''
    (g,l,r) = (-1,-1,-1)
    output = []
    for j in range(len(seq_gestures)):
        if (g != seq_gestures[j]):
            if (g != NO_GESTURE):
                output += [[g, l*FRAMES_PER_CLIP, (r+1)*FRAMES_PER_CLIP-seq_paddings[j]]]  
            g = seq_gestures[j]
            l = j
            r = j

    output_file = open(output_csv_file_path, 'w')
    for row in output:
        output_file.write(repr(int(row[0])) + "," + repr(int(row[1])) + "," + repr(int(row[2])) + "\n")
        output_file.close()




# LOG_DIR = '/home/lc/Dynamic-Mario-Bros/src_ctc/runs/1497347590'
LOG_DIR = '/home/federico/Dynamic-Mario-Bros/src_ctc/runs/1497471639/'
META_GRAPH_FILE = 'model-15500.meta'
# MODEL_CP_PATH = '/home/lc/Dynamic-Mario-Bros/src_ctc/runs/1497347590'
MODEL_CP_PATH = '/home/federico/Dynamic-Mario-Bros/src_ctc/runs/1497471639/'
OUTPUT_PATH = '../evaluation/prediction/'''

tf.flags.DEFINE_string('log_dir', LOG_DIR, 'Checkpoint directory')
tf.flags.DEFINE_string('meta_graph_file', META_GRAPH_FILE, 'Name of meta graph file')
tf.flags.DEFINE_string('model_checkpoint_path', MODEL_CP_PATH, 'Name of checkpoint file')
tf.flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Batch size')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print('\nCommand-line Arguments:')
for attr, value in sorted(FLAGS.__flags.items()):
    print('{}={}'.format(attr.upper(), value))
    print('')

prev_sample = -1
prev_gesture= -1

#with tf.Session(config=tf.ConfigProto(device_count={'GPU':0})) as sess:
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(os.path.join(FLAGS.log_dir, FLAGS.meta_graph_file))
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.log_dir))

    predictions = tf.get_collection('predictions')[0]
    input_samples_op = tf.get_collection('input_samples_op')[0]
    mode = tf.get_collection('mode')[0]

    logits = tf.get_default_graph().get_tensor_by_name("accuracy/Reshape:0")
    logits_soft = tf.nn.softmax(logits)

    for sample_id in tqdm(TEST_ID):
        print('========== sample %d ===========' % sample_id)
        sample = GestureSample('%s/%s/Sample%04d.zip' % (RAW_DATA_PATH, 'Test', sample_id))

        num_of_frames = sample.getNumFrames()
        num_of_clip_batch = math.ceil(num_of_frames/FRAMES_PER_VIDEO/BATCH_SIZE)
        # get entire video
        user = sample.get_entire_user_video()
        vid = sample.get_entire_rgb_video()
        mask = np.mean(user, axis=3) > 150
        mask = mask.reshape((mask.shape + (1,)))
        vid = vid*mask

        results = []
        for clip_range in range(int(num_of_clip_batch)):
            start = clip_range * FRAMES_PER_CLIP * BATCH_SIZE
            end = (clip_range + 1) * FRAMES_PER_CLIP * BATCH_SIZE
            end_padding = 0

            if end > num_of_frames:
                end_padding = end - num_of_frames
                end = num_of_frames

            batch_clips = np.asarray([list(vid[start:end]) + [vid[0]]*end_padding])
            batch_clips = batch_clips[:,:, int((IMAGE_SIZE[0] - CROP[1])/2) : int((IMAGE_SIZE[0] + CROP[1])/2),
                                           int((IMAGE_SIZE[1] - CROP[2])/2) : int((IMAGE_SIZE[1] + CROP[2])/2),:]
            batch_clips = np.asarray(batch_clips, dtype=np.float32).reshape((BATCH_SIZE, FRAMES_PER_CLIP) + (CROP[1], CROP[2], CROP[3]))

            feed_dict = {mode:False, input_samples_op:batch_clips}
            _, logs = sess.run([predictions, logits_soft], feed_dict = feed_dict) #[0].tolist()
            preds = np.argmax(logs, axis=1)
            preds[np.max(logs, axis=1) < 0.4] = NO_GESTURE - 1
            results += preds.tolist()

        #print (results)
        ''' write into files '''
        prev_gesture = results[0]
        start = 0
        gestures = []
        for i in range(len(results)):
            if (results[i] != prev_gesture or (i+1) == len(results)):

                start_frame = start * FRAMES_PER_CLIP + 1

                if ((i+1) == len(results)):
                    end_frame = min((i+1)*FRAMES_PER_CLIP, num_of_frames)
                else:
                    end_frame = min(i * FRAMES_PER_CLIP, num_of_frames)

                if (prev_gesture+1 != NO_GESTURE):
                    gestures += [[prev_gesture+1, start_frame, end_frame]]
                    #gestures += [[prev_gesture, int((start_frame-1)/FRAMES_PER_CLIP), int((end_frame-1)/FRAMES_PER_CLIP)]]

                prev_gesture = results[i]
                start = i
                if (i * FRAMES_PER_CLIP + 1 > num_of_frames):
                    break


        print('results')
        print(results)
        print('gestures')
        print(gestures)

        fout = open('%s/Sample%04d_prediction.csv' % (OUTPUT_PATH, sample_id), 'w')
        for row in gestures:
            fout.write(repr(int(row[0])) + ',' + repr(int(row[1])) + ',' + repr(int(row[2])) + '\n')
        fout.close()




#if __name__ == '__main__':
#
#    tf.app.run()
