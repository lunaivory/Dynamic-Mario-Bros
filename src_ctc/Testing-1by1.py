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
from scipy import stats

import util_testing


LOG_DIR = '/home/lc/Dynamic-Mario-Bros/src_ctc/runs/1497460159'
#LOG_DIR = '/home/federico/Dynamic-Mario-Bros/src_ctc/runs/1497450063'
META_GRAPH_FILE = 'model-2500.meta'
# MODEL_CP_PATH = '/home/lc/Dynamic-Mario-Bros/src_ctc/runs/1497347590'
MODEL_CP_PATH = '/home/lc/Dynamic-Mario-Bros/src_ctc/runs/1497460159'
OUTPUT_PATH = '../evaluation/prediction/'


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

with tf.Session(config=tf.ConfigProto(device_count={'GPU':1})) as sess:
#with tf.Session() as sess:
    saver = tf.train.import_meta_graph(os.path.join(FLAGS.log_dir, FLAGS.meta_graph_file))
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.log_dir))

    predictions = tf.get_collection('predictions')[0]
    input_samples_op = tf.get_collection('input_samples_op')[0]
    mode = tf.get_collection('mode')[0]

    logits = tf.get_default_graph().get_tensor_by_name("accuracy/Reshape:0")
    logits_soft = tf.nn.softmax(logits)

    raw_results = []

    for sample_id in tqdm(TEST_ID):
        print('========== sample %d ===========' % sample_id)
        sample = GestureSample('%s/%s/Sample%04d.zip' % (RAW_DATA_PATH, 'Test', sample_id))

        num_of_frames = sample.getNumFrames()
        num_of_clip_batch = math.ceil(num_of_frames/BATCH_SIZE)
        # get entire video
        user = sample.get_entire_user_video()
        vid = sample.get_entire_rgb_video()
        mask = np.mean(user, axis=3) > 150
        mask = mask.reshape((mask.shape + (1,)))
        vid = vid*mask

        results = []
        for clip_range in range(int(num_of_clip_batch)):
            
            batch_clips = []
            for i in range(BATCH_SIZE):
                start = clip_range * BATCH_SIZE - FRAMES_PER_CLIP + i
                end = clip_range * BATCH_SIZE + i
                start_padding = 0
                end_padding = 0

                if (start < 0):
                    start_padding = -start
                    start = 0

                if start >= num_of_frames:
                    start = num_of_frames
                    end = num_of_frames
                    end_padding = FRAMES_PER_CLIP
                elif end > num_of_frames:
                    end_padding = end - num_of_frames
                    end = num_of_frames

#                print(start, end, start_padding, end_padding, num_of_frames)
                batch_clips += [np.asarray([vid[0]] * start_padding + list(vid[start:end]) + [vid[end-1]]*end_padding).reshape([FRAMES_PER_CLIP, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]])]

            batch_clips = np.asarray(batch_clips)
#            start = clip_range * BATCH_SIZE
#            end = (clip_range + 1) * BATCH_SIZE
#            end_padding = 0
#
#            if end > num_of_frames:
#                end_padding = end - num_of_frames
#                end = num_of_frames
#
#            batch_clips = np.asarray([list(vid[start:end]) + [vid[end-1]]*end_padding])
#            print(batch_clips.shape)
            batch_clips = batch_clips[:,:, int((IMAGE_SIZE[0] - CROP[1])/2) : int((IMAGE_SIZE[0] + CROP[1])/2),
                                           int((IMAGE_SIZE[1] - CROP[2])/2) : int((IMAGE_SIZE[1] + CROP[2])/2),:]
            batch_clips = np.asarray(batch_clips, dtype=np.uint8).reshape((BATCH_SIZE, FRAMES_PER_CLIP) + (CROP[1], CROP[2], CROP[3]))

            feed_dict = {mode:False, input_samples_op:batch_clips}
            _, logs = sess.run([predictions, logits_soft], feed_dict = feed_dict) #[0].tolist()
            preds = np.argmax(logs, axis=1)
            preds[np.max(logs, axis=1) < 0.5] = NO_GESTURE - 1

            results += preds.tolist()

        raw_results += [results]

        #print (results)
        ''' write into files '''
        gestures = []


        ''' cheating version'''
        prev_gestures = []
        start = -1

        for i in range(len(results)):
            results[i]+=1
            if (results[i] == NO_GESTURE and len(prev_gestures) != 0):
                gestures += [[stats.mode(prev_gestures)[0][0], start+1, i]]
                prev_gestures = []
                start = -1
            if (results[i] != NO_GESTURE):
                prev_gestures += [results[i]]
                start = i if start == -1 else start

        print(results)
        print(gestures)
            
#
#
#
#        ''' normal version'''
#        prev_gesture = results[0]
#        start = 0
#        for i in range(len(results)):
#            if (results[i] != prev_gesture or (i+1) == len(results)):
#
#                start_frame = start * FRAMES_PER_CLIP + 1
#
#                if ((i+1) == len(results)):
#                    end_frame = min((i+1)*FRAMES_PER_CLIP, num_of_frames)
#                else:
#                    end_frame = min(i * FRAMES_PER_CLIP, num_of_frames)
#
#                if (prev_gesture+1 != NO_GESTURE):
#                    gestures += [[prev_gesture+1, start_frame, end_frame]]
#                    #gestures += [[prev_gesture, int((start_frame-1)/FRAMES_PER_CLIP), int((end_frame-1)/FRAMES_PER_CLIP)]]
#
#                prev_gesture = results[i]
#                start = i
#                if (i * FRAMES_PER_CLIP + 1 > num_of_frames):
#                    break


##        print('results')
##        print(results)
##        print('gestures')
##        print(gestures)
##
##        input("Press Enter to continue...")

        fout = open('%s/Sample%04d_prediction.csv' % (OUTPUT_PATH, sample_id), 'w')
        for row in gestures:
            if (row[2] - row[1] < 20):
                continue
            fout.write(repr(int(row[0])) + ',' + repr(int(row[1])) + ',' + repr(int(row[2])) + '\n')
        fout.close()

#    fout = open('%s/%s.csv' % (OUTPUT_PATH, META_GRAPH_FILE.split('.')[0]), 'w')
#    for row in raw_results:
#        fout.write(row)
#        fout.write('\n')
#    fout.close()




#if __name__ == '__main__':
#
#    tf.app.run()
