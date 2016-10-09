#!/usr/bin/env python

import rospy

import roslib
roslib.load_manifest('harp_msgs')
from harp_msgs.srv import *

from cv_bridge import CvBridge, CvBridgeError
import cv_bridge
import cv2

from utils.get_test_data import *
from utils.picking_perception_utils import *


class item_identification:
    def __init__(self, model, weights):
        self.net = ID_net(model, weights, 0)
        self.output_img = []
        self.labeled_img = []
        self.do_pub = 0
        self.bridge = CvBridge()
        return

    def get_image(self, data):
        image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        return image

    def predict_scene(self, input_image, bin):
        
        # Note image is in BGR format
        image = self.get_image(input_image)
        
        # Read bin contents
        bin_contents = bin.bin_contents
        bin_target = bin.target_item
        possibleItems, targetItem = get_item_list_ros(bin_contents, bin_target)

        # Run prediction algorithm
        output_mask_list, score_list = identify_image(image, possibleItems, -1, self.net)
        
        # Return results
        output_image_list = []
        for image in output_mask_list:
            output_image_list.append(self.bridge.cv2_to_imgmsg(image, "bgr8"))
        
        return output_image_list, score_list