#!/usr/bin/env python

import sys
import rospy
from harp_msgs.srv import *
from harp_msgs.msg import *

import os
import pdb
import time

import cv2
from cv_bridge import CvBridge, CvBridgeError

import matplotlib.pyplot as plt
import numpy

from sensor_msgs.msg import Image

from utils.get_test_data import *
from utils.picking_perception_utils import *
from utils.item_identification import *


import rospkg

if __name__ == "__main__":

    rospy.init_node('perception_test', anonymous=True)
    
    # Load ROS parameters from vision_config.yaml 
    num_2_item_dict = rospy.get_param("num_2_item_dict")
    save_resulting_images = rospy.get_param("save_resulting_images")
    dataset_path = rospy.get_param("dataset_path")
    
    # Initialize CNN
    rospack = rospkg.RosPack()
    model = rospack.get_path('harp_perception') + rospy.get_param("alexnet_prototxt")  
    weights = rospack.get_path('harp_perception') + rospy.get_param("alexnet_weights")    
    object_recognition = item_identification(model, weights)

    # Wait for geometry filter server
    print "TEST initialized, waiting for geometry filter server"
    rospy.wait_for_service('geometry_filter_server')
    geometry_filter_srv = rospy.ServiceProxy('geometry_filter_server', geometryFilterData)

    # CVbridge converts images to/from ROS messages
    bridge = CvBridge()

    # Open text files to save results
    # NOTE these will be overwritten every time
    cnn_results = open(dataset_path + "cnn_results.txt", "w")
    cnn_results.write ("input,target_item,target_number,num_items,IOU,accuracy,score,run_time\n")

    # Make directory to save results
    if (save_resulting_images):
        if not os.path.isdir(dataset_path+"cnn_results"):
            os.makedirs(dataset_path+"cnn_results")

    # Load file image list 
    hd_img_source = dataset_path+"gtimg/"
    filecount = 0
    dirlist = os.listdir (hd_img_source)

    # Run test on all files in list
    for filenum in range(len(dirlist)):

        # Read file name
        filename = dirlist[filenum]
        filecount = filecount + 1
        print "RUNNING ALGO ON FILE # %d" % (filenum)
        print "File Name: %s" % (filename)
        hd_img_file = filename
        hd_img_path = hd_img_source+hd_img_file
        hd_img = cv2.imread(hd_img_path)
        fileStr = hd_img_file.split("_")[0]

        # Load mask file, to determine possible items
        ground_truth_mask_source = hd_img_source.replace("gtimg", "gtmask")
        ground_truth_mask_file = hd_img_file
        ground_truth_mask = cv2.imread(ground_truth_mask_source+ground_truth_mask_file)

        # Get possible item list generated from file name
        possibleItems = get_item_list([ground_truth_mask_file])
        possibleItems = possibleItems[0]
        possibleItems = [x+1 for x in possibleItems]
        paddedPossibleItems = possibleItems + [0]*(12 - len(possibleItems))
        
        # Generate bin contents message
        bin_in = bin()
        bin_in.name = "bin_A"
        bin_in.num_items = np.count_nonzero(paddedPossibleItems)
        bin_in.bin_contents = paddedPossibleItems

        # Start timer
        t0 = time.time()

        # Create bin contents message
        bin_in.target_item = possibleItems[0]
        
        # Run geometry based filter
        print "Filtering based on geometry only"
        #req = geometryFilterDataRequest()
        #req.file_name = fileStr
        #req.run_test = 1
        #res = geometry_filter_srv(req)
        #cnn_mask = res.img_out
        req = geometryFilterDataRequest()
        req.cloud_path = ground_truth_mask_source.replace("gtmask", "hdcloud") + fileStr + "_hdcloud.pcd"
        req.image_path = ground_truth_mask_source.replace("gtmask", "hdmask")  + fileStr + "_maskimg.jpg"
        req.tf_path = ground_truth_mask_source.replace("gtmask", "tfdata") + fileStr + "_tfdata.txt"
        res = geometry_filter_srv(req)
        cnn_mask = res.img_out

        # Run item identification
        print 'Calling ID CNN, predicting scene'
        output_image_list, scores = object_recognition.predict_scene(cnn_mask, bin_in)

        # Stop timer
        t1 = time.time()

        # Calcluating resutls on all items
        for i in range (len(scores)):

            item = possibleItems[i]
            itemName = num_2_item_dict[str(item)]
            print 'Saving Results for item %s' % (itemName)

            item_id = output_image_list[i]
            item_id = bridge.imgmsg_to_cv2(item_id, "bgr8")

            runtime = t1 - t0
            fileStr = hd_img_file.split("_")[0]

            if (save_resulting_images):
                cnn_result_source = hd_img_source.replace("gtimg", "cnn_results")
                cnn_results_path = cnn_result_source + fileStr + "_" + "target" + str(item) + ".jpg"
                alpha = .9
                identification_result = cv2.addWeighted(item_id, alpha, hd_img, 1 - alpha, 0)
                alpha = .75
                image_mask = bridge.imgmsg_to_cv2(cnn_mask, "bgr8")
                identification_result = cv2.addWeighted(identification_result, alpha, image_mask, 1 - alpha, 0)
                cv2.imwrite(cnn_results_path, identification_result) 

            item_id_output = np.copy(item_id)

            id_index = (item_id_output != 0)
            item_id_output[id_index] = 1

            item_id_output=item_id_output[:,:,0]

            IOU = getIOU(item_id_output, ground_truth_mask, item)
            accuracy = getAccuracy(item_id_output, ground_truth_mask, item)
            
            numItems = bin_in.num_items

            score = scores[i]

            cnn_results.write ("%s,%s,%d,%d,%4f,%4f,%4f,%4f\n" % (fileStr, itemName, item, numItems,IOU,accuracy,score,runtime))

    print "Test finished, saving text files"
    cnn_results.close()

