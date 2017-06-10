
import tensorflow as tf
import numpy as np
import model
import matplotlib.pyplot as plt
import time as time
import skvideo.io
import os

'''####################################################'''
'''#                 Self-defined library             #'''
'''####################################################'''
from constants import *
import util_training
import util_val

def main(argv):
    graph = tf.Graph()

    with graph.as_default():

        ''' place holders'''
        global_step = tf.Variable(1, name='global_step', trainable=False)
        mode = tf.placeholder(dtype=tf.bool, name='mode') # Pass True in when it is in the trainging mode

        input_samples_op, input_labels_op, input_dense_label_op, input_clip_label_op = util_training.input_pipeline(TRAIN_FILENAMES)

        input_seq_op = [BATCH_SIZE]

        '''Define the cells'''
        logits = model.dynamic_mario_bros(input_samples_op, DROPOUT_RATE, mode)

        # loss = tf.nn.ctc_loss(input_labels_op, logits, input_seq_op, time_major=False)

        # learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step,
        # decay_steps = 1 * FLAGS.epoch_length, decay_rate=0.96, stiarecase=True)
        # optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        # train_op = optimizer.minimize(loss, global_step=global_step)

    with tf.Session(graph=graph, config=tf.ConfigProto(device_count={'GPU':0})) as sess:
    # with tf.Session(graph=graph) as sess:
        # set up the queue

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:

            for epoch in range(1, NUM_EPOCHS + 1):

                if coord.should_stop():
                    break

                train_cost = 0
                start = time.time()

                for batch in range(10): #range(round(len(TRAIN_FILENAMES) / BATCH_SIZE)):

                        feed_dict = {mode: True}
                        input_samples, input_labels, input_clip_labels = sess.run([input_samples_op, input_labels_op, input_clip_label_op], feed_dict)
                        name = "train%02d.mp4" % (batch)
                        temp  =np.reshape(input_samples, (FRAMES_PER_VIDEO*BATCH_SIZE, 112, 112, 3))
                        skvideo.io.vwrite(name, temp)
                        print('Saved video ' + name)

        except Exception as e:
            # Report exceptions to the coordinator.
            print(str(e))
            coord.request_stop(e)

        finally:
        # Terminate as usual. It is safe to call `coord.request_stop()` twice.
            coord.request_stop()
            coord.join(threads)


    # train_cost /= len(TRAIN_FILENAMES)
    # print('[%d/%d] [Training] Loss : %.3f' % (epoch, epoch * len(TRAIN_FILENAMES)/BATCH_SIZE + batch, train_loss))

if __name__ == '__main__':

    tf.app.run(main)
