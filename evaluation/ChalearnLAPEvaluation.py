#-------------------------------------------------------------------------------
# Name:        Chalearn LAP evaluation scripts
# Purpose:     Provide evaluation scripts for Chalearn LAP challenge tracks
#
# Author:      Xavier Baro
#              Miguel Angel Bautista
#
# Created:     21/01/2014
# Copyright:   (c) Chalearn LAP 2014
# Licence:     GPL
#-------------------------------------------------------------------------------
import os
import zipfile
import shutil
import re
import csv
import numpy
from PIL import Image


def gesture_overlap_csv(csvpathgt, csvpathpred, seqlenght):
    """ Evaluate this sample agains the ground truth file """
    maxGestures=20

    # Get the list of gestures from the ground truth and frame activation
    gtGestures = []
    binvec_gt = numpy.zeros((maxGestures, seqlenght))
    with open(csvpathgt, 'r') as csvfilegt:
        csvgt = csv.reader(csvfilegt)
        for row in csvgt:
            binvec_gt[int(row[0])-1, int(row[1])-1:int(row[2])-1] = 1
            gtGestures.append(int(row[0]))

    # Get the list of gestures from prediction and frame activation
    predGestures = []
    binvec_pred = numpy.zeros((maxGestures, seqlenght))
    with open(csvpathpred, 'r') as csvfilepred:
        csvpred = csv.reader(csvfilepred)
        for row in csvpred:
            binvec_pred[int(row[0])-1, int(row[1])-1:int(row[2])-1] = 1
            predGestures.append(int(row[0]))

    # Get the list of gestures without repetitions for ground truth and predicton
    gtGestures = numpy.unique(gtGestures)
    predGestures = numpy.unique(predGestures)

    # Find false positives
    falsePos=numpy.setdiff1d(gtGestures, numpy.union1d(gtGestures,predGestures))

    # Get overlaps for each gesture
    overlaps = []
    for idx in gtGestures:
        intersec = sum(binvec_gt[idx-1] * binvec_pred[idx-1])
        aux = binvec_gt[idx-1] + binvec_pred[idx-1]
        union = sum(aux > 0)
        overlaps.append(intersec/union)

    # Use real gestures and false positive gestures to calculate the final score
    return sum(overlaps)/(len(overlaps)+len(falsePos))


def exportGT_Gesture(dataPath, outputPath):
    """ Create Ground Truth folder. Open each file in the data path and copy labels and sample data to output path"""
    # Check the given data path
    if not os.path.exists(dataPath) or not os.path.isdir(dataPath):
        raise Exception("Data path does not exist: " + dataPath)

    # Check the output path
    if os.path.exists(outputPath) and os.path.isdir(outputPath):
        raise Exception("Output path already exists. Remove it before start: " + outputPath)

    # Create the output path
    os.makedirs(outputPath)
    if not os.path.exists(outputPath) or not os.path.isdir(outputPath):
        raise Exception("Cannot create the output path: " + outputPath)

    # Get the list of samples
    samplesList = os.listdir(dataPath)

    # For each sample on the GT, search the given prediction
    for sample in samplesList:
        # Build paths for sample
        sampleFile = os.path.join(dataPath, sample)

        # Check that is a ZIP file
        if not os.path.isfile(sampleFile) or not sample.lower().endswith(".zip"):
            continue

        # Prepare sample information
        file = os.path.split(sampleFile)[1]
        sampleID = os.path.splitext(file)[0]
        samplePath = dataPath + os.path.sep + sampleID

        # Unzip sample if it is necessary
        if os.path.isdir(samplePath):
            unziped = False
        else:
            unziped = True
            zipFile = zipfile.ZipFile(sampleFile, "r")
            zipFile.extractall(samplePath)

        # Copy labels file
        sampleDataPath = samplePath + os.path.sep + sampleID + '_data.csv'
        if not os.path.exists(sampleDataPath):
            raise Exception("Invalid sample file. Sample data is not available")
        shutil.copyfile(sampleDataPath, outputPath + sampleID + '_data.csv')

        # Copy Data file
        srcSampleDataPath = samplePath + os.path.sep + sampleID + '_data.csv'
        dstSampleDataPath = outputPath + os.path.sep + sampleID + '_data.csv'
        if not os.path.exists(srcSampleDataPath) or not os.path.isfile(srcSampleDataPath):
            raise Exception("Invalid sample file. Sample data is not available")
        shutil.copyfile(srcSampleDataPath, dstSampleDataPath)
        if not os.path.exists(dstSampleDataPath) or not os.path.isfile(dstSampleDataPath):
            raise Exception("Cannot copy data file: " + srcSampleDataPath + "->" + dstSampleDataPath)

        # Copy labels file
        srcSampleLabelsPath = samplePath + os.path.sep + sampleID + '_labels.csv'
        dstSampleLabelsPath = outputPath + os.path.sep + sampleID + '_labels.csv'
        if not os.path.exists(srcSampleLabelsPath) or not os.path.isfile(srcSampleLabelsPath):
            raise Exception("Invalid sample file. Sample labels is not available")
        shutil.copyfile(srcSampleLabelsPath, dstSampleLabelsPath)
        if not os.path.exists(dstSampleLabelsPath) or not os.path.isfile(dstSampleLabelsPath):
            raise Exception("Cannot copy labels file: " + srcSampleLabelsPath + "->" + dstSampleLabelsPath)

        # Remove temporal data
        if unziped:
            shutil.rmtree(samplePath)



def evalGesture(prediction_dir,truth_dir):
    """ Perform the overlap evaluation for a set of samples """
    worseVal=10000

    # Get the list of samples from ground truth
    gold_list = sorted(os.listdir(truth_dir))

    # For each sample on the GT, search the given prediction
    numSamples=0.0;
    score=0.0;
    for gold in gold_list:
        # Avoid double check, use only labels file
        if not gold.lower().endswith("_labels.csv"):
            continue

        # Build paths for prediction and ground truth files
        sampleID=re.sub('\_labels.csv$', '', gold)
        labelsFile = os.path.join(truth_dir, sampleID + "_labels.csv")
        dataFile = os.path.join(truth_dir, sampleID + "_data.csv")
        predFile = os.path.join(prediction_dir, sampleID + "_prediction.csv")

        # Get the number of frames for this sample
        with open(dataFile, 'r') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                numFrames=int(row[0])
            del filereader

        # Get the score
        numSamples+=1
        score+=gesture_overlap_csv(labelsFile, predFile, numFrames)

    return score/numSamples
