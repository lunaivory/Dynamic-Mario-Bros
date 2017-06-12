import tensorflow as tf
import numpy as np
import model
import matplotlib.pyplot as plt
import time as time
import os

import util_testing

'''####################################################'''
'''#                Self-defined library              #'''
'''####################################################'''
from constants import *
from sklearn.metrics import jaccard_similarity_score

def exportPredictions(prediction_list, output_csv_file_path, seq_gestures, seq_paddings):
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





META_GRAPH_FILE = LOG_DIR + ''

tf.flags.DEFINE_string('log_dir', LOG_DIR, 'Checkpoint directory')
tf.flags.DEFINE_string('meta_graph_file', META_GRAPH_FILE, 'Name of meta graph file')
tf.flags.DEFINE_string('model_checkpoint_path', LOG_DIR, 'Name of checkpoint file')
tf.flags.DEFINE_interger('batch_size', BATCH_SIZE, 'Batch size')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print('\nCommand-line Arguments:')
for attr, value in sorted(FLAGS.__flags.items()):
    print('{}={}'.format(attr.upper(), value))
    print('')


batch_samples_op, batch_sample_id_op, batch_clip_id_op, batch_padding_op = util_testing.input_pipeline(TEST_FILENAMES)

prev_sample = -1
prev_gesture= -1
with tf.Session() as sess:
    try:
        saver = tf.train.import_meta_graph(os.path.join(FLAGS.log_dir, FLAGS.meta_graph_file))
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.log_dir)

        predictions = tf.get_collections('predictions')[0]
        input_samples_op = tf.get_collection('input_data')[0]
        mode_op = tf.get_collection('mode')[0]

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        feed_dict = {mode:False, input_samples_op:batch_samples_op}
        # or input_samples_op = batch_samples_op
        request_output = [predictions, batch_sample_id_op, batch_clip_id_op, batch_padding_op]
        predictions, sample_id, clip_id, padding = sess.run(requeset_output, feed_dict=feed_dict)



        seq_gestures = []
        seq_paddings = []
        for i in range(len(predictions)):

            if (prev_sample != sample_id[i]):
                if (prev_sample != -1):
                    output_csv_file = os.path.join(prediction_dir,  subjectID + '_prediction.csv')
                    exportPredictions(prediction_list, output_csv_file, seq_gestures, seq_paddings)
                    #write to file
                    seq_gestures = []
                    seq_paddings = []

                # The queue repeat
                if (prev_sample > sample_id[i]):
                    return 
                prev_sample = sample_id[i]

            if clip_id[i] != len(seq_gestures):
                print("Clip ID is not in order!")
                return

            seq_gestures += [predictions[i]]
            seq_paddings += [padding[i]]
            

    except Exception as e:
        # Report exceptions to the coordinator.
        print(str(e))
        coord.request_stop(e)

    finally:
        # Terminate as usual. It is safe to call `coord.request_stop()` twice.
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':

    tf.app.run()
