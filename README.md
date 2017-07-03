# Program pipline
----------------------
1. Put original data under /data/Train, /data/Test, /data/Validation
2. Run ./src_ctc/video_preorocess.sh to create downsized samples,
...which will be store under corresponding directory under /data_pp
3. Run /src_ctc/Raw_data_to_TFRecord.py to create corresponding tfrecords 
...files for training 3DCNN and CNN-RNN network.
4. Run /src_ctc/Training_3dcnn.py to pre-trained the 3DCNN network on
...shuffled clips.
5. Run /srs_ctc/Training.py to trained the RNN using the trained 3DCNN network
...in Step 4, stored under /srs_ctc/runs/.
6. Run Testing_3dcnn.py to test the result from trained 3DCNN network from step 4.
...Run Testing.py to test the result from trained CNN-RNN network from step 5.
...These two files will create .csv file under /evaluation/prediction for each 
...sample under data_pp/Test. After this process, /evaluation/evaluate.py can be
...run to get the jaccard score of the result.

# Our Result
---------------------
1. 3DCNN model reached 90% accuracy in training.
2. RNN-CNN model reached 30% accuracy in training, but can reach 90% accuracy on
...smaller dataset.
3. 3DCNN model reached X% jaccard accuracy on testing dataset.
4. CNN-RNN model reached X% jaccard accuracy on testing dataset.


# File explaination
---------------------
* preprocessing
  * /src_ctc/video_preprocess.sh : downscaled video using bash script.
  * /src_ctc/Raw_data_to_TFRecords.py : master file to create tfrecord files.
    * /src_ctc/training_data_tfrecords_3dcnn_last.py: create tfrecord files for
    training 3DCNN model.
    * /src_ctc/training_data_tfrecords_3dcnn.py: create tfrecord files for training
    RNN_CNN model.
    * Both files ensures that the data has exact same amount of samples in all labels.
* training
  * /src_ctc/Training_3dcnn.py : train the 3DCNN model.
  * /src_ctc/Training.py : load the trained CNN model and train the CNN-RNN model.
  * /src_ctc/util_training_3dcnn.py : input pipeline and preprocessing for training
  3DCNN model.
  * /src_ctc/util_training.py : input pipeline and preprocessing for training
  CNN-RNN model.
  * /src_ctc/constants.py and /src_ctc/constants_3dcnn.py : parameters shares among files.
  * /src_ctc/model.py : definition of the 3DCNN network.
  * /src_ctc/runs : models trained will be record under this directory
* testing
  * /src_ctc/Testing_3dcnn.py : test the 3DCNN model
  * /src_ctc/Testing.py : test the CNN-RNN model



