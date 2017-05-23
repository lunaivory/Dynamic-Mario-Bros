import numpy as np
import pickle
import os
from scipy.misc import imresize
from ChalearnLAPSample import GestureSample
from ChalearnLAPSample import Skeleton
import tensorflow as tf

# TODO: Convert camelcase to underscore style.
# Methods that use tensorflow follows underscore style.

def _int64_feature(value):
  """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
  """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
  return tf.train.Feature(int64_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
  """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))

def _int64_feature_list(values):
  """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

def _bytes_feature_list(values):
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def _float_feature_list(values):
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_float_feature(v) for v in values])

class ChalearnSequenceDataSet(object):
    """
        Class that allows to create a dataset object by providing filtering
        functionality.
    """
    def __init__ (self, config, savePath, dataFolder, serializeLabels=True, labelFilePathFormat=None, sampleNameFormat="Sample{0:04d}.zip"):
        """
            @param config is a dictionary describing details of the dataset object, and
            consisting of the following fields:
            'gestureList': Gestures to be fetched.
            'subjectList': Subjects to be fetched.
            'numSubjectsInTFRecord': # of subjects in a TFRecord file.
            'TFRecordFileStartID': ID of the first file. Useful if you do multiple runs.
            'crop': Borders of crop area (left, right, top, bottom).
            'resizeRatio': Resize ratio.

            @param savePath: Location of output TFRecord files.
            @param dataFolder: Location of ChaLearn data (subject zip files)
            @param serializeLabels: Create "label" field in TFRecords.
            @param labelFilePathFormat: Useful if the labels are not located in subject zip files.
            @param sampleNameFormat

            (1) Initialize the object.
            (2) Call readData() method.
        """
        # Check the given folder
        if not os.path.exists(dataFolder): #or not os.path.isfile(fileName):
            raise Exception("Data folder does not exist: " + dataFolder)

        self.config = config
        self.dataFolder = dataFolder
        self.serializeLabels = serializeLabels
        self.labelFilePathFormat = labelFilePathFormat
        self.sampleNameFormat = sampleNameFormat
        self.savePath = savePath

        self.numSamples = 0
        self.numBlankSamples = 0
        self.data = {}
        self.data['rgb'] = []
        self.data['segmentation'] = []
        self.data['depth'] = []
        self.data['skeleton'] = []
        self.gestureLabel = 0

        self.tfRecordFileID = config['TFRecordFileStartID']
        self.numSubjectsInTFRecord = 0

        # Create first record file
        filename = os.path.join(self.savePath + '_' + str(self.tfRecordFileID) + '.tfrecords')
        print('Writing', filename)
        self.tfWriterOptions = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)
        self.tFRecordWriter = tf.python_io.TFRecordWriter(filename, options=self.tfWriterOptions)
        self.tfRecordFileID += 1

    def make_tf_record(self):
        """Converts a sample into tf.SequenceExample."""
        if self.serializeLabels:
            context = tf.train.Features(feature={
                "label": _int64_feature(self.gestureLabel),
                "length": _int64_feature(len(self.data['rgb'])),
            })
        else:
            context = tf.train.Features(feature={
                "length": _int64_feature(len(self.data['rgb'])),
            })

        featureLists = tf.train.FeatureLists(feature_list={
            "depth": _bytes_feature_list(self.data['depth']),
            "rgb": _bytes_feature_list(self.data['rgb']),
            "segmentation": _bytes_feature_list(self.data['segmentation']),
            "skeleton": _bytes_feature_list(self.data['skeleton']),
        })
        sequenceExample = tf.train.SequenceExample(
            context=context, feature_lists=featureLists)
        return sequenceExample

    def write_tf_record(self):
        """
        Creates a tf.SequenceExample from current sample and writes it into
        tf.Record
        """
        if self.numSubjectsInTFRecord >= self.config['numSubjectsInTFRecord']:
            self.tFRecordWriter.close()
            filename = os.path.join(self.savePath + '_' + str(self.tfRecordFileID) + '.tfrecords')
            print('Writing', filename)
            self.tFRecordWriter = tf.python_io.TFRecordWriter(filename, options=self.tfWriterOptions)
            self.tfRecordFileID += 1
            self.numSubjectsInTFRecord = 0

        sequenceExample = self.make_tf_record()
        self.tFRecordWriter.write(sequenceExample.SerializeToString())

    def finalize_tf_record_writing(self):
        self.tFRecordWriter.close()

    def readData(self):
        """
        Iterate over all subjects.
        """
        counter = 1
        step = round(len(self.config['subjectList'])/10)
        if step == 0:
            step = len(self.config['subjectList'])

        for subjectID in self.config['subjectList']:
            if counter % step == 0:
                print(str(counter)+"/"+str(len(self.config['subjectList'])))
            numSubjectSamples = self.readSubjectData(subjectID)
            counter = counter + 1
            self.numSubjectsInTFRecord += 1
            print("Subject: " + str(subjectID) + " # Samples: " + str(numSubjectSamples))

        self.finalize_tf_record_writing()
        print("# Samples: " + str(self.numSamples))
        print("# Blank Samples: " + str(self.numBlankSamples))

    def readSubjectData(self, subjectID):
        """
        Creates gesture object, reads subject data, creates gesture clips and
        writes into TFRecord.
        """
        if self.labelFilePathFormat is not None:
            gestureObject = GestureSample(self.dataFolder + self.sampleNameFormat.format(subjectID), 
                self.dataFolder+'/labels/'+self.labelFilePathFormat)
        else:
            gestureObject = GestureSample(self.dataFolder + self.sampleNameFormat.format(subjectID))
        labels = gestureObject.getGestures()
        # Filter gesture labels.
        numSubjectSamples = 0
        for labelEntry in labels: # labelEntry = [gestureLabel, startFrame, endFrame]
            if labelEntry[0] in self.config['gestureList']:
                try:
                    # If the clip has blank segmentation mask, discard it.
                    if self.fetchFrames(gestureObject, range(labelEntry[1], labelEntry[2])):
                        self.gestureLabel = labelEntry[0] #s.append(np.ones((1), dtype='uint8')*labelEntry[0])
                        self. ()
                        self.numSamples += 1
                        numSubjectSamples += 1
                except:
                    print(str(labelEntry[1]) + " - " + str(labelEntry[2]))

                self.data['rgb'] = []
                self.data['segmentation'] = []
                self.data['depth'] = []
                self.data['skeleton'] = []
                self.gestureLabel = 0
        return numSubjectSamples

    def fetchFrames(self, gestureObject, frameList):
        """
        ChalearnLAPSample API provides functions for reading data per frame.
        Creates a list of gesture frames, and merge them.
        """
        for f in frameList:
            segmentationImg = gestureObject.getUser(f)
            # If the segmentation is blank, then discard it.
            if segmentationImg.sum() == 0:
                self.numBlankSamples += 1
                return False
            self.data['segmentation'].append(self.tailorFrames(segmentationImg))
            self.data['rgb'].append(self.tailorFrames(gestureObject.getRGB(f)))
            self.data['depth'].append(self.tailorFrames(gestureObject.getDepth(f)))
            self.data['skeleton'].append(np.expand_dims(gestureObject.getSkeleton(f).getRawData(), 0))
        return True

    def tailorFrames(self, frame):
        """
        Crops, resizes the images based on config.
        """
        if self.config['crop'] is not None:
            frame = frame[self.config['crop'][0]:self.config['crop'][1], self.config['crop'][2]:self.config['crop'][3]]
        if self.config['resizeRatio'] > 0:
            frame = imresize(frame, self.config['resizeRatio'], interp='bilinear')

        return np.expand_dims(frame, 0)
