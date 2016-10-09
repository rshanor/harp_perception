#!/usr/bin/env python

import rospy 

caffe_root = rospy.get_param("caffe_root") 
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

import numpy as np
import numpy.matlib
from opencv_utils import *

from sklearn.preprocessing import normalize

from picking_perception_utils import *

import math
import time


class ID_net:
    ## The Constructor.
    # Initializes the CNN
    def __init__(self, model, weights, GPU):
        # Caffe Settings
        caffe.set_mode_gpu()
        caffe.set_device(GPU)
        self.net = caffe.Classifier(model,weights)
        self.transformer = self.get_transformer()
        from sklearn.preprocessing import normalize

    ## Caffe tool to transform image into proper input
    def get_transformer(self):
        # Set network transformer
        transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_channel_swap('data', (2,1,0))
        transformer.set_raw_scale('data', 255.0)
        return transformer

    # Preprosses data and makes prediction on single iamge
    def do_forward(self, image):
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', image)
        out = self.net.forward()
        predictions = out['prob']
        return predictions

    # Set non possible item predictions to 0
    # For items not in possible item list
    def filter_predictions(self, predictions, possibleItems):
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=3)
        # Set impossible predictions to 0
        finalPred = np.zeros(predictions.shape)
        for num in possibleItems:
            finalPred[0,num] = predictions[0,num]
        rawPred = finalPred[0,:]
        normPred =  rawPred / math.sqrt(np.sum(np.square(rawPred)))
        return rawPred, normPred

    # Loading image from openCV required format change
    def convert_to_caffe(self, image_file):
        img_init = image_file
        img = img_init.astype(np.float32)
        img = img / 255
        input_image = swap_channels (img)
        return input_image

    # Classify all non black superpixels in an image
    def classifySuperpixels(self, image, segments, possibleItems):
        # Initialize arrays that hold results for each segment
        output_mask = np.zeros(image.shape[:2], dtype = "uint8")
        count = 1
        imageArray = []
        segmentArray = []
        rawPredictionArray = []
        normPredictionArray = []

        # Iterate over each superpixel
        for (i, segVal) in enumerate(np.unique(segments)):
            t0 = time.time()
            # construct a mask for the segment
            tempmask = np.zeros(image.shape[:2], dtype = "uint8")
            tempmask[segments == segVal] = 255

            masked_image = cv2.bitwise_and(image, image, mask = tempmask)
            nonBlack = cv2.countNonZero(masked_image[:,:,1])
            maskSize = cv2.countNonZero(tempmask)

            # Only look at superpixels that are large and not black 
            if (nonBlack*1.0/maskSize > .60 and nonBlack > 1200):
                # Create pixel-based labeled clusters
                
                from picking_perception_utils import *

                segment_crop, output_mask = getSuperpixel(image,
                                                        segments,
                                                        segVal,
                                                        output_mask,
                                                        masked_image,
                                                        count)
                count = count+1

                imageArray.append(masked_image)
                segmentArray.append(segment_crop)

                # Run prediction on item
                # Forward calculate network
                net_input = self.convert_to_caffe(segment_crop)
                t1 = time.time()
                predictions = self.do_forward(net_input)
                t2 = time.time()
                raw_prediction, norm_prediction = self.filter_predictions(predictions, 
                                                                        possibleItems)
                rawPredictionArray.append(raw_prediction)
                normPredictionArray.append(norm_prediction)

                #print "Segment ID time: %5f seconds" % (t2-t1)
            t3 = time.time()
            #print "Total ID time: %5f seconds" % (t3-t0)

        return output_mask, imageArray, segmentArray, rawPredictionArray, normPredictionArray

