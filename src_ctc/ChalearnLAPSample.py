#-------------------------------------------------------------------------------
# Name:        Chalearn LAP sample
# Purpose:     Provide easy access to Chalearn LAP challenge data samples
#
# Author:      Xavier Baro
# Created:     21/01/2014
# Author:      Lionel Pigou: lionelpigou@gmail.com
# Modified:    11/10/2014
# Author: eaksan
# Modified:    8/2/2017 (Converted from python2 to python3)
#-------------------------------------------------------------------------------
"""
from ChalearnLAPSample3 import GestureSample
gestureSample = GestureSample('<PATH_TO_DATA>/Sample<ID>.zip' )
"""
import os
import zipfile
import shutil
import cv2
from numpy import *
import csv
from PIL import Image, ImageDraw
from scipy.misc import imresize
import numpy
from scipy import misc

class Skeleton(object):
    """ Class that represents the skeleton information """
    #define a class to encode skeleton data
    def __init__(self,data):
        """ Constructor. Reads skeleton information from given raw data """
        self.rawData = numpy.array(data, dtype=numpy.float32)
        # Create an object from raw data
        self.joins=dict();
        pos=0
        self.joins['HipCenter']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['Spine']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['ShoulderCenter']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['Head']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['ShoulderLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['ElbowLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['WristLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['HandLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['ShoulderRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['ElbowRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['WristRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['HandRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['HipLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['KneeLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['AnkleLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['FootLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['HipRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['KneeRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['AnkleRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['FootRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
    def getRawData(self):
        """ Returns all the information as numpy array which can be used to construct a skeleton object later. """
        return self.rawData

    def getAllData(self):
        """ Return a dictionary with all the information for each skeleton node """
        return self.joins
    def getWorldCoordinates(self):
        """ Get World coordinates for each skeleton node """
        skel=dict()
        for key in list(self.joins.keys()):
            skel[key]=self.joins[key][0]
        return skel
    def getJoinOrientations(self):
        """ Get orientations of all skeleton nodes """
        skel=dict()
        for key in list(self.joins.keys()):
            skel[key]=self.joins[key][1]
        return skel
    def getPixelCoordinates(self):
        """ Get Pixel coordinates for each skeleton node """
        skel=dict()
        for key in list(self.joins.keys()):
            skel[key]=self.joins[key][2]
        return skel
    def toImage(self,width,height,bgColor):
        """ Create an image for the skeleton information """
        SkeletonConnectionMap = (['HipCenter','Spine'],['Spine','ShoulderCenter'],['ShoulderCenter','Head'],['ShoulderCenter','ShoulderLeft'], \
                                 ['ShoulderLeft','ElbowLeft'],['ElbowLeft','WristLeft'],['WristLeft','HandLeft'],['ShoulderCenter','ShoulderRight'], \
                                 ['ShoulderRight','ElbowRight'],['ElbowRight','WristRight'],['WristRight','HandRight'],['HipCenter','HipRight'], \
                                 ['HipRight','KneeRight'],['KneeRight','AnkleRight'],['AnkleRight','FootRight'],['HipCenter','HipLeft'], \
                                 ['HipLeft','KneeLeft'],['KneeLeft','AnkleLeft'],['AnkleLeft','FootLeft'])
        im = Image.new('RGB', (width, height), bgColor)
        draw = ImageDraw.Draw(im)
        for link in SkeletonConnectionMap:
            p=self.getPixelCoordinates()[link[1]]
            p.extend(self.getPixelCoordinates()[link[0]])
            draw.line(p, fill=(255,0,0), width=5)
        for node in list(self.getPixelCoordinates().keys()):
            p=self.getPixelCoordinates()[node]
            r=5
            draw.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r),fill=(0,0,255))
        del draw
        image = numpy.array(im)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image


class GestureSample(object):
    """ Class that allows to access all the information for a certain gesture database sample """
    #define class to access gesture data samples
    def __init__ (self,fileName,labelFileName=None):
        """ Constructor. Read the sample file and unzip it if it is necessary. All the data is loaded.

            sample=GestureSample('Sample0001.zip')

        """
        # Check the given file
        if not os.path.exists(fileName): #or not os.path.isfile(fileName):
            raise Exception("Sample path does not exist: " + fileName)

        # Prepare sample information
        self.fullFile = fileName
        self.dataPath = os.path.split(fileName)[0]
        self.file=os.path.split(fileName)[1]
        self.seqID=os.path.splitext(self.file)[0]
        self.samplePath=self.dataPath + os.path.sep + self.seqID;

        # Unzip sample if it is necessary
        if os.path.isdir(self.samplePath) :
            self.unzip = False
        else:
            self.unzip = True
            zipFile=zipfile.ZipFile(self.fullFile,"r")
            zipFile.extractall(self.samplePath)

        # Open video access for RGB information
        rgbVideoPath=self.samplePath + os.path.sep + self.seqID + '_color.mp4'
        if not os.path.exists(rgbVideoPath):
            raise Exception("Invalid sample file. RGB data is not available")

        self.rgb = cv2.VideoCapture(rgbVideoPath)

        while not self.rgb.isOpened():
            self.rgb = cv2.VideoCapture(rgbVideoPath)
            cv2.waitKey(500)
            # Open video access for Depth information
        depthVideoPath=self.samplePath + os.path.sep + self.seqID + '_depth.mp4'

        if not os.path.exists(depthVideoPath):
            raise Exception("Invalid sample file. Depth data is not available")
        self.depth = cv2.VideoCapture(depthVideoPath)

        while not self.depth.isOpened():
            self.depth = cv2.VideoCapture(depthVideoPath)
            cv2.waitKey(500)
            # Open video access for User segmentation information
        userVideoPath=self.samplePath + os.path.sep + self.seqID + '_user.mp4'
        if not os.path.exists(userVideoPath):
            raise Exception("Invalid sample file. User segmentation data is not available")
        self.user = cv2.VideoCapture(userVideoPath)
        while not self.user.isOpened():
            self.user = cv2.VideoCapture(userVideoPath)
            cv2.waitKey(500)
            # Read skeleton data
        skeletonPath=self.samplePath + os.path.sep + self.seqID + '_skeleton.csv'
        if not os.path.exists(skeletonPath):
            raise Exception("Invalid sample file. Skeleton data is not available")
        self.skeletons=[]
        with open(skeletonPath, 'rt') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                self.skeletons.append(Skeleton(row))
            del filereader
            # Read sample data
        sampleDataPath=self.samplePath + os.path.sep + self.seqID + '_data.csv'
        if not os.path.exists(sampleDataPath):
            raise Exception("Invalid sample file. Sample data is not available")
        self.data=dict()
        with open(sampleDataPath, 'rt') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                self.data['numFrames']=int(row[0])
                self.data['fps']=int(row[1])
                self.data['maxDepth']=int(row[2])
            del filereader

        # Read labels data
        self.labels=[]
        labelsPath=self.samplePath + os.path.sep + self.seqID + '_labels.csv'
        if not os.path.exists(labelsPath):
            labelsPath = None

        if labelFileName is not None:
            labelsPath = labelFileName

        if labelsPath is not None:
            with open(labelsPath, 'rt') as csvfile:
                filereader = csv.reader(csvfile, delimiter=',')
                for row in filereader:
                    self.labels.append(list(map(int,row)))
                del filereader

    def __del__(self):
        """ Destructor. If the object unziped the sample, it remove the temporal data """
        if self.unzip:
            self.clean()

    def clean(self):
        """ Clean temporal unziped data """
        del self.rgb;
        del self.depth;
        del self.user;
        shutil.rmtree(self.samplePath)

    def getFrame(self,video, frameNum):
        """ Get a single frame from given video object """
        # Check frame number
        # Get total number of frames
        numFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        # Check the given file
        if frameNum<1 or frameNum>numFrames:
            raise Exception("Invalid frame number <" + str(frameNum) + ">. Valid frames are values between 1 and " + str(int(numFrames)))
            # Set the frame index
        video.set(cv2.CAP_PROP_POS_FRAMES,frameNum-1)
        ret,frame=video.read()
        if ret==False:
            raise Exception("Cannot read the frame")
        return frame

    def getRGB(self, frameNum):
        """ Get the RGB color image for the given frame """
        #get RGB frame
        return self.getFrame(self.rgb,frameNum)

    def getDepth(self, frameNum):
        """ Get the depth image for the given frame """
        #get Depth frame
        depthData=self.getFrame(self.depth,frameNum)
        # Convert to grayscale
        depthGray=cv2.cvtColor(depthData,cv2.COLOR_RGB2GRAY)
        # Convert to float point
        #depth=depthGray.astype(numpy.float32)
        ## Convert to depth values
        #depth=depth/255.0*float(self.data['maxDepth'])
        #depth=depth.round()
        #depth=depth.astype(numpy.uint16)
        return depthGray

    def getUser(self, frameNum):
        """ Get user segmentation image for the given frame """
        #get user segmentation frame
        return self.getFrame(self.user,frameNum)

    def getSkeleton(self, frameNum):
        """ Get the skeleton information for a given frame. It returns a Skeleton object """
        #get user skeleton for a given frame
        # Check frame number
        # Get total number of frames
        numFrames = len(self.skeletons)
        # Check the given file
        if frameNum<1 or frameNum>numFrames:
            raise Exception("Invalid frame number <" + str(frameNum) + ">. Valid frames are values between 1 and " + str(int(numFrames)))
        return self.skeletons[frameNum-1]

    def getSkeletonImage(self, frameNum):
        """ Create an image with the skeleton image for a given frame """
        return self.getSkeleton(frameNum).toImage(640,480,(255,255,255))

    def getNumFrames(self):
        """ Get the number of frames for this sample """
        return self.data['numFrames']

    def getComposedFrame(self, frameNum):
        """ Get a composition of all the modalities for a given frame """
        # get sample modalities
        rgb=self.getRGB(frameNum)
        depthValues=self.getDepth(frameNum)
        user=self.getUser(frameNum)
        skel=self.getSkeletonImage(frameNum)

        # Build depth image
        depth = depthValues.astype(numpy.float32)
        depth = depth*255.0/float(self.data['maxDepth'])
        depth = depth.round()
        depth = depth.astype(numpy.uint8)
        depth = cv2.applyColorMap(depth,cv2.COLORMAP_JET)

        # Build final image
        compSize1=(max(rgb.shape[0],depth.shape[0]),rgb.shape[1]+depth.shape[1])
        compSize2=(max(user.shape[0],skel.shape[0]),user.shape[1]+skel.shape[1])
        comp = numpy.zeros((compSize1[0]+ compSize2[0],max(compSize1[1],compSize2[1]),3), numpy.uint8)

        # Create composition
        comp[:rgb.shape[0],:rgb.shape[1],:]=rgb
        comp[:depth.shape[0],rgb.shape[1]:rgb.shape[1]+depth.shape[1],:]=depth
        comp[compSize1[0]:compSize1[0]+user.shape[0],:user.shape[1],:]=user
        comp[compSize1[0]:compSize1[0]+skel.shape[0],user.shape[1]:user.shape[1]+skel.shape[1],:]=skel

        return comp

    def getComposedFrameOverlapUser(self, frameNum):
        """ Get a composition of all the modalities for a given frame """
        # get sample modalities
        rgb=self.getRGB(frameNum)
        depthValues=self.getDepth(frameNum)
        user=self.getUser(frameNum)
        mask = numpy.mean(user, axis=2) > 150
        mask = numpy.tile(mask, (3,1,1))
        mask = mask.transpose((1,2,0))

        # Build depth image
        depth = depthValues.astype(numpy.float32)
        depth = depth*255.0/float(self.data['maxDepth'])
        depth = depth.round()
        depth = depth.astype(numpy.uint8)
        depth = cv2.applyColorMap(depth,cv2.COLORMAP_JET)

        # Build final image
        compSize=(max(rgb.shape[0],depth.shape[0]),rgb.shape[1]+depth.shape[1])
        comp = numpy.zeros((compSize[0]+ compSize[0],max(compSize[1],compSize[1]),3), numpy.uint8)

        # Create composition
        comp[:rgb.shape[0],:rgb.shape[1],:]=rgb
        comp[:depth.shape[0],rgb.shape[1]:rgb.shape[1]+depth.shape[1],:]= depth
        comp[compSize[0]:compSize[0]+user.shape[0],:user.shape[1],:]= mask * rgb
        comp[compSize[0]:compSize[0]+user.shape[0],user.shape[1]:user.shape[1]+user.shape[1],:]= mask * depth

        return comp

    def getComposedFrame_480(self, frameNum, ratio=0.5, topCut=60, botCut=140):
        """ Get a composition of all the modalities for a given frame """
        # get sample modalities

        rgb=self.getRGB(frameNum)
        rgb = rgb[topCut:-topCut,botCut:-botCut,:]

        rgb = imresize(rgb, ratio, interp='bilinear')
        depthValues=self.getDepth(frameNum)
        user=self.getUser(frameNum)
        user = user[topCut:-topCut,botCut:-botCut,:]
        user = imresize(user, ratio, interp='bilinear')
        mask = numpy.mean(user, axis=2) > 150
        mask = numpy.tile(mask, (3,1,1))
        mask = mask.transpose((1,2,0))

        # Build depth image
        depth = depthValues.astype(numpy.float32)
        depth = depth*255.0/float(self.data['maxDepth'])
        depth = depth.round()
        depth = depth[topCut:-topCut,botCut:-botCut]
        depth = imresize(depth, ratio, interp='bilinear')
        depth = depth.astype(numpy.uint8)
        depth = cv2.applyColorMap(depth,cv2.COLORMAP_JET)

        # Build final image
        compSize=(max(rgb.shape[0],depth.shape[0]),rgb.shape[1]+depth.shape[1])
        comp = numpy.zeros((compSize[0]+ compSize[0],max(compSize[1],compSize[1]),3), numpy.uint8)

        # Create composition
        comp[:rgb.shape[0],:rgb.shape[1],:]=rgb
        comp[:depth.shape[0],rgb.shape[1]:rgb.shape[1]+depth.shape[1],:]= depth
        comp[compSize[0]:compSize[0]+user.shape[0],:user.shape[1],:]= mask * rgb
        comp[compSize[0]:compSize[0]+user.shape[0],user.shape[1]:user.shape[1]+user.shape[1],:]= mask * depth

        return comp

    def getDepth3DCNN(self, frameNum, ratio=0.5, topCut=60, botCut=140):
        """ Get a composition of all the modalities for a given frame """
        # get sample modalities

        depthValues=self.getDepth(frameNum)
        user=self.getUser(frameNum)
        user = user[topCut:-topCut,botCut:-botCut,:]
        user = imresize(user, ratio, interp='bilinear')
        mask = numpy.mean(user, axis=2) > 150

        # Build depth image
        depth = depthValues.astype(numpy.float32)
        depth = depth*255.0/float(self.data['maxDepth'])
        depth = depth.round()
        depth = depth[topCut:-topCut,botCut:-botCut]
        depth = imresize(depth, ratio, interp='bilinear')
        depth = depth.astype(numpy.uint8)

        return  mask * depth

    def getDepthOverlapUser(self, frameNum, x_centre, y_centre, pixel_value, extractedFrameSize=224, upshift = 0):
        """ Get a composition of all the modalities for a given frame """
        halfFrameSize = extractedFrameSize/2
        user=self.getUser(frameNum)
        mask = numpy.mean(user, axis=2) > 150

        ratio = pixel_value/ 3000

        # Build depth image
        # get sample modalities
        depthValues=self.getDepth(frameNum)
        depth = depthValues.astype(numpy.float32)
        depth = depth*255.0/float(self.data['maxDepth'])

        mask = imresize(mask, ratio, interp='nearest')
        depth = imresize(depth, ratio, interp='bilinear')

        depth_temp = depth * mask
        depth_extracted = depth_temp[x_centre-halfFrameSize-upshift:x_centre+halfFrameSize-upshift, y_centre-halfFrameSize: y_centre+halfFrameSize]

        depth = depth.round()
        depth = depth.astype(numpy.uint8)
        depth = cv2.applyColorMap(depth,cv2.COLORMAP_JET)

        depth_extracted = depth_extracted.round()
        depth_extracted = depth_extracted.astype(numpy.uint8)
        depth_extracted = cv2.applyColorMap(depth_extracted,cv2.COLORMAP_JET)

        # Build final image
        compSize=(depth.shape[0],depth.shape[1])
        comp = numpy.zeros((compSize[0] + extractedFrameSize,compSize[1]+compSize[1],3), numpy.uint8)

        # Create composition
        comp[:depth.shape[0],:depth.shape[1],:]=depth
        mask_new = numpy.tile(mask, (3,1,1))
        mask_new = mask_new.transpose((1,2,0))
        comp[:depth.shape[0],depth.shape[1]:depth.shape[1]+depth.shape[1],:]= mask_new * depth
        comp[compSize[0]:,:extractedFrameSize,:]= depth_extracted

        return comp

    def getDepthCentroid(self, startFrame, endFrame):
        """ Get a composition of all the modalities for a given frame """
        x_centre = []
        y_centre = []
        pixel_value = []

        for frameNum in range(startFrame, endFrame):
            user=self.getUser(frameNum)
            depthValues=self.getDepth(frameNum)
            depth = depthValues.astype(numpy.float32)
            #depth = depth*255.0/float(self.data['maxDepth'])
            mask = numpy.mean(user, axis=2) > 150
            width, height = mask.shape
            XX, YY, count, pixel_sum = 0, 0, 0, 0
            for x in range(width):
                for y in range(height):
                    if mask[x, y]:
                        XX += x
                        YY += y
                        count += 1
                        pixel_sum += depth[x, y]
            if count>0:
                x_centre.append(XX/count)
                y_centre.append(YY/count)
                pixel_value.append(pixel_sum/count)

        return [numpy.mean(x_centre), numpy.mean(y_centre), numpy.mean(pixel_value)]

    def getGestures(self):
        """ Get the list of gesture for this sample. Each row is a gesture, with the format (gestureID,startFrame,endFrame) """
        return self.labels

    def getGestureName(self,gestureID):
        """ Get the gesture label from a given gesture ID """
        names=('vattene*','vieniqui','perfetto','furbo*','cheduepalle','chevuoi','daccordo','seipazzo', \
               'combinato','freganiente*','ok*','cosatifarei*','basta','prendere*','noncenepiu*','fame*','tantotempo', \
               'buonissimo','messidaccordo','sonostufo*')
        # Check the given file
        if gestureID<1 or gestureID>20:
            raise Exception("Invalid gesture ID <" + str(gestureID) + ">. Valid IDs are values between 1 and 20")
        return names[gestureID-1]

    def exportPredictions(self, prediction, predPath):
        """ Export the given prediction to the correct file in the given predictions path """
        if not os.path.exists(predPath):
            os.makedirs(predPath)
        output_filename = os.path.join(predPath,  self.seqID + '_prediction.csv')
        output_file = open(output_filename, 'w')
        for row in prediction:
            output_file.write(repr(int(row[0])) + "," + repr(int(row[1])) + "," + repr(int(row[2])) + "\n")
        output_file.close()

    def evaluate(self,csvpathpred):
        """ Evaluate this sample agains the ground truth file """
        maxGestures=11
        seqLength=self.getNumFrames()

        # Get the list of gestures from the ground truth and frame activation
        predGestures = []
        binvec_pred = numpy.zeros((maxGestures, seqLength))
        gtGestures = []
        binvec_gt = numpy.zeros((maxGestures, seqLength))
        with open(csvpathpred, 'rt') as csvfilegt:
            csvgt = csv.reader(csvfilegt)
            for row in csvgt:
                binvec_pred[int(row[0])-1, int(row[1])-1:int(row[2])-1] = 1
                predGestures.append(int(row[0]))

        # Get the list of gestures from prediction and frame activation
        for row in self.getActions():
                binvec_gt[int(row[0])-1, int(row[1])-1:int(row[2])-1] = 1
                gtGestures.append(int(row[0]))

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
