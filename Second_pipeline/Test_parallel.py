import numpy as np
import cv2
import matplotlib.pyplot
from matplotlib import pyplot as plt

def denoise(left,right,resize_shape,window_size,sigma):
    '''Return the denoised gray-scale version of left and right images.
    window_size and sigma are parameters for Gaussian blur'''
    #make sure the color is correct
    left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
    left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)  # shape:(2000, 2964, 1)
    right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)  # shape:(2000, 2964, 1)

    # Resize image to increase the speed of analysis
    left_resized = cv2.resize(left, resize_shape)  # shape:(500, 741, 1)
    right_resized = cv2.resize(right, resize_shape)
    # smooth 2 images with 2D Gaussian filter
    left_blur = cv2.GaussianBlur(left_resized, window_size, sigma)
    right_blur = cv2.GaussianBlur(right_resized, window_size, sigma)
    return left_blur,right_blur


if __name__ == '__main__':
    disparity = cv2.imread("../Images/parallel/disparity_15.png")
    plt.imshow(disparity)
    plt.show()
    # The distance between the two cameras, taken from calib.txt
    T = 1438.004 - 1263.818
    # The focal length of the two cameras, taken from calib.txt
    f = 5299.313

    depth_map = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if disparity[i,j] == 0:
                depth_map[i,j] = 255
            else:
                depth_map[i,j] = (f*T)/disparity[i,j]
    plt.imshow(depth_map)
    plt.show()
    cv2.imwrite('../Images/parallel/depth_15.png', depth_map)








