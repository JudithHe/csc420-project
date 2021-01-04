"""
PFM Python implementation used to load the ground truth depth map files provided with ShinySMVS as a numpy array
PFM code segment originally developed by Dr.Yao: https://github.com/YoYo000/MVSNet
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import cv2  # OpenCV2

def load_pfm(file):
    color = None
    width = None
    height = None
    scale = None
    data_type = None
    print("Load PFM file " + str(file))
    try:
        header = file.readline().decode('UTF-8').rstrip()
    except:
        print("Invalid header file")

    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('UTF-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
        print("PFM width=%d and height=%d" % (width, height))
    else:
        raise Exception('Malformed PFM header.')
    # scale = float(file.readline().rstrip())
    scale = float((file.readline()).decode('UTF-8').rstrip())
    if scale < 0: # little-endian
        data_type = '<f'
    else:
        data_type = '>f' # big-endian
    data_string = file.read()
    data = np.fromstring(data_string, data_type)
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = cv2.flip(data, 0)
    return data

if __name__ == '__main__':
    # Loading the PFM file
    depth_image = load_pfm(open("00000006.pfm", 'rb'))
    print('value range: ', depth_image.min(), depth_image.max())
    plt.imshow(depth_image, 'rainbow')
    plt.show()
