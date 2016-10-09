# This file contains configuration information
# for testing the scene identification approach

import numpy as np
import os
import rospy

## Documentation for a class.
#
# Image data struct contains info about an image.
# Used only when running on test data.
class image_data:
    ## The constructor
    def __init__(self, folder):
        self.image_list, self.mask_list = get_image_list(folder)
        self.item_list = get_item_list(self.image_list)
        self.numbers = getNumbers()
        self.names = getNames()

## Documentation for a function.
# 
# This function generates the list of file names and their masks
def get_image_list(folder):
    image_list = []
    mask_list = []
    for fn in os.listdir(folder):
        image_folder = folder
        mask_folder = folder.replace("images","masks")
        image_list.append(image_folder + fn)
        mask_list.append(mask_folder + fn)
    return image_list, mask_list

def get_item_list_ros(item_list, target):
    numbers = getNumbers()
    image_item_list = []
    for item in item_list:
        if (item != 0):
            itemNumToIndex = numbers.index(str(item))
            image_item_list.append(itemNumToIndex)
    target_item = numbers.index(str(target))
    return image_item_list, target_item

## Documentation for a function.
#
# Items are encoded into the image file name
# This function parses the filename to generate possible and taraget item
def get_item_list(image_list):
    item_list = []
    numbers = getNumbers()
    for image in image_list:
        image_item_list = []
        image_split = image.split("-")
        for text in image_split:
            if (text.isdigit() and len(text)<=2):
                itemNumToIndex = numbers.index(text)
                image_item_list.append(itemNumToIndex)
        item_list.append(image_item_list)
    return item_list  

def getNumbers():
    names = getNames()
    numbers = []
    for i in range (len(names)):
        numbers.append(str(i+1))
    return numbers

# Return the names of the items in a list
def getNames():
    nameList = []
    num_2_item_dict = rospy.get_param("num_2_item_dict")
    for i in range (len(num_2_item_dict)):
        item = num_2_item_dict[str(i+1)]
        nameList.append(item)

    return nameList