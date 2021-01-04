'''Stereo Parallel Version'''
import numpy as np
import cv2
from matplotlib import pyplot as plt


#Checking time: first time

def zeroPad(img, h_pad, w_pad):
    '''Return a zero padded image with new shape
    row = img.row+h_pad
    col = img.col+w_pad'''
    img_h,img_w = img.shape
    new_h = img_h + h_pad
    new_w = img_w + w_pad
    new_img = np.zeros((new_h, new_w), 'uint8')

    # copy orginal image into new_img with zero padding
    n = 0
    for i in range(h_pad//2, new_h - h_pad//2):
        m = 0
        for j in range(w_pad//2, new_w - w_pad//2):
            new_img[i, j] = img[n, m]
            m = m + 1
        n = n + 1
    print('padded img has shape,',new_img.shape)
    return new_img

def getValue(padded_img, x, y,patch_size,h_pad,w_pad):
    '''Return the array of elements in the patch of the padded image
    x,y is the center pixel of the window in the image without padding.
    Parameter: h_pad, w_pad: added rows and columns
    '''
    #the coordinate in the new padded image
    x_pad,y_pad = x+h_pad//2,y+w_pad//2
    side_len = patch_size[0]//2

    result = np.zeros(patch_size[0]*patch_size[0]) #shape: patch_size:5x5
    n = 0
    for i in range(x_pad-side_len, x_pad+side_len+1):
        for j in range(y_pad-side_len,y_pad+side_len+1):
            #print('information,', i,j)
            result[n] = padded_img[i,j]
            n += 1
    return result

def disparity_SSD(padded_imgl,padded_imgr,rowi,coli,patch_size,h_pad,w_pad):
    '''Return the disparity "xl-xr" given a point(rowi,coli) in left image(imgl)
    using SSD(sum of Square difference)
    rowi: row index of left image
    coli: col index of left image'''
    left_patch = getValue(padded_imgl, rowi,coli, patch_size, h_pad, w_pad)

    minimum_SSD = float('inf')
    #Consider all columns in right image with the same row index(rowi)
    #Only search columns within (0,xl)
    if coli == 0:
        #on the leftmost column of the image
        disparity = 0
    for col in range(coli):
        right_patch = getValue(padded_imgr, rowi, col, patch_size, h_pad, w_pad)
        #Use Gray images Or add color channel
        SSD = np.sum(np.square(np.subtract(left_patch,right_patch)))
        if SSD < minimum_SSD:
            minimum_SSD = SSD
            #disparity = xl-xr
            disparity = coli-col
    return disparity

def Depthmap(imgl,imgr, f, T,patch_size,h_pad,w_pad):
    '''Return the depth of each pixel in 2D image plane in
    the camera coordinate system(Z_c) and the disparity map of a stereo pair'''
    rows,cols = imgl.shape
    depth_map = np.zeros((rows,cols))
    disparity_map = np.zeros((rows, cols))
    # zero padding for 2 images
    padded_imgl = zeroPad(imgl, h_pad, w_pad)
    padded_imgr = zeroPad(imgr, h_pad, w_pad)

    for i in range(rows):
        for j in range(cols):
            result = disparity_SSD(padded_imgl,padded_imgr,i,j,patch_size,h_pad,w_pad)
            #print('disparity is ,', result)
            disparity_map[i,j] = result
            if result == 0:
                depth_map[i,j] = 255
            else:
                depth_map[i,j] = (f*T)/result
    #put depth into range of 0-255
    maxi_depth = np.max(depth_map)
    mini_depth = np.min(depth_map)
    depth_map = ((depth_map-mini_depth)/(maxi_depth-mini_depth))*255
    return disparity_map, depth_map

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


# The distance between the two camera centers
T = 1438.004 - 1263.818

# The focal length of the two cameras, taken from calib.txt
focal_length = 5299.313

#Read 2 parallel images
left = cv2.imread("../Test_Images/parallel/left.png")
right = cv2.imread("../Test_Images/parallel/right.png")
#left_blur and right blur has shape of 200x300
left_blur,right_blur = denoise(left, right, (300, 200),(5,5), 2)
plt.title('Left image(Blurred grayscale version)')
plt.imshow(left_blur, cmap='gray')
plt.show()
plt.title('Right image(Blurred grayscale version)')
plt.imshow(right_blur, cmap='gray')
plt.show()

#result = disparity_SSD(left_resized,right_resized, 0, 0, (5,5), 10,10)
patch_size = (5,5)
h_pad = 10
w_pad = 10
disparity_map,depth_map = Depthmap(left_blur,right_blur, focal_length, T,patch_size,h_pad,w_pad)
plt.title('Disparity map')
plt.imshow(disparity_map)
plt.title('Depth map')
plt.show()
plt.imshow(depth_map)
plt.show()
cv2.imwrite('../Final_results/Second_pipeline/disparity_5.png', disparity_map)
cv2.imwrite('../Final_results/Second_pipeline/depth_map_5.png', depth_map)








