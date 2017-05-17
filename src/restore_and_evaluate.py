import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import os
import utils

#Get test data
path = '/home/fzechini/Desktop/uie2/data/testData_pp_rgb_masked.pkl'

# Pass directory of your model.
tf.flags.DEFINE_string("log_dir","/home/fzechini/Desktop/uie2/skeleton/runs/1493114062", "Checkpoint directory")
# Pass name of meta-graph-file in <log_dir>
tf.flags.DEFINE_string("meta_graph_file", "model-28000.meta", "Name of meta graph file")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size (default: 32)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nCommand-line Arguments:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

model_id = FLAGS.log_dir.split('/')[-1]
if model_id == '':
    model_id = FLAGS.log_dir.split('/')[-2]

def main(unused_argv):
    # Load test data.
    test_data = utils.get_test_data(path)
    #test_data = test_data.reshape(-1, 90, 90, 3)

    with tf.Session() as sess:
        # Restore computation graph.
        saver = tf.train.import_meta_graph(os.path.join(FLAGS.log_dir, FLAGS.meta_graph_file))
        # Restore variables.
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.log_dir))
        # Restore ops.
        predictions = tf.get_collection('predictions')[0]
        input_samples_op = tf.get_collection('input_samples_op')[0]
        mode = tf.get_collection('mode')[0]

        def do_prediction(sess, samples):
            batches = utils.data_iterator_samples(samples, FLAGS.batch_size)
            test_predictions = []
            for batch_samples in batches:
                feed_dict = {input_samples_op: batch_samples,
                             mode: False}
                test_predictions.extend(sess.run(predictions, feed_dict=feed_dict))
            return test_predictions

        test_predictions = do_prediction(sess, test_data)
        # Create submission file.
        with open('prediction_final.csv', 'w') as outcsv:
            outcsv.write('Id,Prediction\n')
            for (i, item) in enumerate(test_predictions):
                outcsv.write(str(i + 1) + ',' + str(item) + '\n')

if __name__ == '__main__':
    tf.app.run()
