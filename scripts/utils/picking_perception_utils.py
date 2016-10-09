#!/usr/bin/env python

import cv2

import rospy

import numpy as np

import warnings
import pdb

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from sklearn import cluster

import time

from pylab import *
from scipy.ndimage import measurements

from get_test_data import *
from opencv_utils import *
from caffe_utils import *
from visualization_utils import *

# Graph data structure
class Segment():
    def __init__(self, cx, cy, neighbors, image, raw, norm):
        self.cx = cx
        self.cy = cy
        self.neighbors = neighbors
        self.image = image
        self.raw_prediction = raw
        self.norm_prediction = norm

# Returns the full image and the 256x256 image of a given superpixel
def getSuperpixel(image,
                  segments, 
                  segVal,
                  output_mask,
                  masked_image,
                  count):
    segmask = np.zeros(image.shape[:2], dtype = "uint8")
    segmask[segments == segVal] = count
    output_mask = output_mask|segmask 
    # Save masked image, for later network training
    # Find COM, crop to 256x256 region
    M = cv2.moments(masked_image[:,:,0])
    s = 256
    cx, cy = findCOM(masked_image,s)
    segment_crop = masked_image[cy-s/2:cy+s/2,cx-s/2:cx+s/2,:]
    return segment_crop,output_mask        

# Runs SLIC on a given image
def do_slic(num_segments, comp, image):
    # Supress warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        segments = slic(img_as_float(image),
                        n_segments = num_segments, 
                        sigma = 0,
                        compactness = comp,
                        convert2lab=True)
    return segments

# Finds neighboring superpixels for a given superpixel
def getNeighbors(image, output_mask, superpic_val):
    # Generate mask around segment i
    segment_mask = np.zeros(image.shape[:2], dtype = "uint8")
    segment_mask[output_mask == superpic_val] = 10 

    # DST is a mask, with a 5X5 kernel applied, so segment should slightly grow
    kernel = np.ones((5,5),np.float32)
    dst = cv2.filter2D(segment_mask,-1,kernel)
    dst = np.minimum(np.ones(image.shape[:2], dtype = "uint8"), dst)
    dst = dst * 255

    # Apply mask to segment mask
    superpix_mask = cv2.bitwise_and(output_mask, output_mask, mask = dst)
    # get neighbors
    neighbors = np.unique(superpix_mask)
    neighbors = np.trim_zeros(neighbors)
    neighbors = np.setdiff1d(neighbors, np.array([superpic_val]))
    neighbors = np.subtract(neighbors, 1)

    M = cv2.moments(segment_mask)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    return neighbors,cx,cy

# Generates a graph of all neighboring superpixels
def generateGraph(image, output_mask, imageArray, rawPredictionArray, normPredictionArray):
    num_segments = np.amax(output_mask)
    graph = []
    # Create graph
    for i in range (num_segments):
        superpic_val = i+1

        neighbors,cx,cy = getNeighbors(image, 
                                 output_mask, 
                                 superpic_val)
        # Add graph to segment
        seg = Segment(cx, cy, 
                      neighbors,
                      imageArray[i],
                      rawPredictionArray[i],
                      normPredictionArray[i])
        graph.append(seg)

    return graph

# Overlay graph on top of image
def drawGraph(image, graph):
    graph_image = np.copy(image)
    for seg in graph:
        for n in seg.neighbors:
            cx1 = seg.cx
            cy1 = seg.cy
            index = n
            seg2 = graph[index]
            cx2 = seg2.cx
            cy2 = seg2.cy
            cv2.line(graph_image, (cx1,cy1), (cx2,cy2),(255, 0, 0),2)
    return graph_image


def globalID (image, graph, possibleItems):
    
    remainingItems = list(possibleItems)

    neighbor_weight = rospy.get_param("neighbor_weight")

    return_images = [0]*len(getNames())

    # Possible graph nodes to still label 
    remainingNodeIndexes = list(xrange(len(graph)))

    if (len(graph) == 0 and len(possibleItems) == 1):
        return_images[possibleItems[0]] = image
        return return_images
    # Error Handling
    elif (len(graph) == 0):
        return -1

    while len(remainingItems) > 0:
        
        best_score = 0
        best_node = []
        best_item = 0
        best_node_index = 0
        # Find absolute best score in graph
        for item in remainingItems: 
            for nodeIndex in remainingNodeIndexes:
                node = graph[nodeIndex]
                score = node.raw_prediction[item]

                for neighbor_index in node.neighbors:
                    neighbor = graph[neighbor_index]
                    neighbor_score = neighbor.raw_prediction[item]
                    score += neighbor_weight * neighbor_score
                if (score > best_score):
                    best_node = node
                    best_score = score
                    best_item = item
                    best_node_index = nodeIndex

        if (best_score == 0):
            return -1

        remainingItems.remove(best_item)
        
        identified_nodes = [best_node_index]

        success_threshold = 1.0/len(possibleItems) - .0001

        print "Target Item %d" % (best_item) 

        for neighbor_index in best_node.neighbors:
            neighbor = graph[neighbor_index]
            score = neighbor.norm_prediction[best_item]
            #print "Neighbor score %f" % (score)
            if (score >= success_threshold) and (remainingNodeIndexes.count(neighbor_index)):
                identified_nodes.append(neighbor_index)

        output_mask = np.zeros(image.shape, dtype = "uint8")
        #print "IDENTIFIED NODES"
        #print identified_nodes
        for success in identified_nodes:
            node_image = graph[success].image
            output_mask = output_mask|node_image
            remainingNodeIndexes.remove(success)

        return_images[best_item] = output_mask

    return return_images

def singleItemID(image, graph, targetItem):

    best_score = 0 
    best_node = []
    ct = 0
    for node in graph:
        score = node.raw_prediction[targetItem]
        #print node.norm_prediction
        #print node.neighbors
        for neighbor_index in node.neighbors:
            neighbor = graph[neighbor_index]
            neighbor_score = neighbor.raw_prediction[targetItem]
            score += .5 * neighbor_score
        ct = ct + 1
        if (score > best_score):
            best_node = node
            best_score = score

    output_mask = np.zeros(image.shape, dtype = "uint8")
    node_image = best_node.image
    output_mask = output_mask|node_image
    for neighbor_index in best_node.neighbors:
        neighbor = graph[neighbor_index]
        node_image = neighbor.image
        output_mask = output_mask|node_image

    return output_mask, best_score, best_node


def getTargetOutput(image, targetItem, out_map):
    # Find item labeled output from CRF
    item_label = np.zeros(out_map.shape,dtype="uint8")
    item_index = (out_map == targetItem)
    item_label[item_index]=1
    # Label segments and calculate max area
    lw, num = measurements.label(np.asarray(item_label))
    area = measurements.sum(item_label, lw, index=arange(lw.max() + 1))
    best_val = np.argmax(area)
    # Mask image, create single segment
    image_mask = np.zeros(out_map.shape,dtype="uint8")
    best_index = (lw == best_val)
    image_mask[best_index] = 1
    image_mask = np.swapaxes(image_mask,0,1)
    #image_mask = np.swapaxes(image_mask,0,1)
    final_mask = np.zeros(image.shape, dtype="uint8")
    # Hack, dont know why repmat wont work...
    final_mask[:,:,0] = image_mask
    final_mask[:,:,1] = image_mask
    final_mask[:,:,2] = image_mask
    final_mask_index = (final_mask == 0)
    image_out = np.copy(image)
    image_out[final_mask_index] = 0

    return image_out

# Main function that runs SLIC, generates a superpixel graph, 
# predicts superpixels, and solves for optimal scene
def identify_image(image, possibleItems, targetItem, net):
    # Set slic parameters, these should be equal during train and test
    num_segments = 1000
    comp = 30

    # Misc. runtime params
    # Use these to visualize various stages of algorithm
    # TODO pull these into ROS param server
    show_slic = 0
    save_superpixels = 0
    show_graph = 0
    show_result = 0

    # Run slic on 
    print "Running SLIC"
    t0 = time.time()
    segments = do_slic(num_segments, comp, image)
    if (show_slic == 1):
        visualize_slic(image, segments)
    t1 = time.time()
    print "Ran SLIC in %3f seconds" % (t1-t0)
    
    # Classify 
    # TODO classify all superpixels at once
    print "Classifying Superpixels"
    t0 = time.time()
    (output_mask, 
    imageArray, 
    segmentArray,
    rawPredictionArray, 
    normPredictionArray) = net.classifySuperpixels(image, 
                                                  segments, 
                                                  possibleItems)
    t1 = time.time()
    print "Ran classification in %3f seconds" % (t1-t0)

    # Generate a graph of connecting superpixels
    print "Generating Graph"
    t0 = time.time()
    graph = generateGraph(image, output_mask, imageArray, rawPredictionArray, normPredictionArray)
    t1 = time.time()
    print "Generated graph in %3f seconds" % (t1-t0)

    if (show_graph == 1):
        graph_image = drawGraph(image, graph)
        plt.axis("off")
        plt.imshow(cv2.cvtColor(graph_image, cv2.COLOR_BGR2RGB))
        plt.show()
    
    print "Identifying Best Segment"
    t0 = time.time()
    numItems = len(getNames())
    scoreArray = []
    imageArray = []

    run_global_optimization = rospy.get_param("run_global_optimization")

    if (targetItem != -1):
        
        identification_result, score, best_node = singleItemID(image, graph, targetItem)
        scoreArray.append(score)    
        imageArray.append(identification_result)        
        t1 = time.time()
        print "Identified items in %3f seconds" % (t1-t0)
        return imageArray,score   
    
    # TESTING WITH GLOBAL OPTIMIZATION, run ID on all items
    elif (run_global_optimization):
        image_list = globalID(image, graph, possibleItems)

        # Error handling
        if (image_list == -1):
            return -1, -1

        for item in possibleItems:
            imageArray.append(image_list[item])

        return imageArray,possibleItems   

    # ELSE TESTING W/O GLOBAL ID, run ID on all items
    for item in possibleItems:
        
        if (item != 0):
            identification_result, score, best_node = singleItemID(image, graph, item)
            scoreArray.append(score)
            # Generate circle on the best point, for testing only
            pt = (best_node.cx, best_node.cy)
            cv2.circle(identification_result, pt, 15, (0,0,255), thickness=-1)
            imageArray.append(identification_result)

    t1 = time.time()
    print "Identified all items in %3f seconds" % (t1-t0)        

    return imageArray, scoreArray


# Intersection over union calculation based on ground truth image
def getIOU (output_image, ground_truth, targetItem):
    
    image_mask = np.copy(output_image) 
    #output_image[:,:,0]
    item_index = (image_mask != 0)
    image_mask[item_index] = 1 

    gt = zeros(ground_truth.shape, dtype="uint8")
    item_index = (ground_truth == targetItem)
    gt[item_index] = 1
    gt = gt[:,:,0]
    #pdb.set_trace()
    # Calculate innersection over union
    overlap = np.count_nonzero(image_mask&gt)
    total = np.count_nonzero(image_mask|gt)
    IOU = overlap*1.0/total
    return IOU    

# Accuracy is defined as 
def getAccuracy (output_image, ground_truth, targetItem):
    
    image_mask = np.copy(output_image) 

    gt = zeros(ground_truth.shape, dtype="uint8")
    item_index = (ground_truth == targetItem)
    gt[item_index] = 1
    gt = gt[:,:,0]

    overlap = np.count_nonzero(image_mask&gt)

    gt2 = zeros(ground_truth.shape, dtype="uint8")
    non_shelf = (ground_truth != 0)
    gt2[non_shelf] = 1
    gt2 = gt2[:,:,0]
    total_overlap = np.count_nonzero(gt2&output_image)
    
    if (total_overlap > 0):
        accuracy = overlap*1.0/total_overlap
    else: accuracy=0

    return accuracy