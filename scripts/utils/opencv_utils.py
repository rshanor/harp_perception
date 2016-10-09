# This file contains openCV tools
# to help with APC scene identification
import cv2
import numpy as np

import pdb

# Finds COM of an image
# s is the desired output image size, 
def findCOM(image,s):
    M = cv2.moments(image[:,:,0])
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])    

    y, x, z = image.shape

    s = s/2
    if (cx < s): cx = s
    if (cy < s): cy = s
    if (cx > (x-s-1)): cx = x-s-1
    if (cy > (y-s-1)): cy = y-s-1

    return cx, cy

# Loading image from openCV required format change
def get_caffe_image(image_file):
    img_init = cv2.imread(image_file)
    # HACK TO FIX POORLY MASKED IMAGES
    # REMOVE THIS LATER
    s = 650
    cx, cy = findCOM(img_init,s)
    temp = np.zeros(img_init.shape, dtype = "uint8")
    temp[cy-s/2:cy+s/2,cx-s/2:cx+s/2,:] = img_init[cy-s/2:cy+s/2,cx-s/2:cx+s/2,:]

    img_init = temp
    img = img_init.astype(np.float32)
    img = img / 255
    input_image = swap_channels (img)
    return img_init, input_image

# Finds the center of an image, ignoring black pixels
def get_image_center(masked_image, mask):
    # Count non black pixels
    nonBlack = cv2.countNonZero(masked_image[:,:,1])
    # Find COM
    M = cv2.moments(mask)
    CX = int(M['m10']/M['m00']) * 4
    CY = int(M['m01']/M['m00']) * 4
    # Find center RGB
    B = np.mean(masked_image[:,:,0])
    G = np.mean(masked_image[:,:,1])
    R = np.mean(masked_image[:,:,2])
    x, y, z = masked_image.shape
    B = B * x * y / nonBlack
    G = G * x * y / nonBlack
    R = R * x * y / nonBlack
    return CX, CY, R, G, B

def swap_channels(img_init):
    red = img_init[:,:,2]
    green = img_init[:,:,1]
    blue = img_init[:,:,0]
    final = cv2.merge([red,green,blue])
    return final

def show_SLIC(input_image, segments):
    fig = plt.figure("Superpixels")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(img_as_float(input_image), segments))
    plt.axis("off")
    plt.show()
    return

def get_color_image(color, size):
    color_image = np.zeros(size, np.uint8)
    color_image[:,:,0] = color[2]
    color_image[:,:,1] = color[1]
    color_image[:,:,2] = color[0]    
    return color_image