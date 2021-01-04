import cv2
import numpy as np
import matplotlib.pyplot
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import scipy.spatial
import scipy.optimize
from scipy.spatial import distance

"""
Referrence:
the main idea in this python file are from the textbook:"Multiple View Geometry in computer vision,
by Richard Hartley and Andrew", a course note:"Epipolar Geometry of Stanford University"
https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf and the website
http://www.sci.utah.edu/~gerig/CS6320-S2012/Materials/CS6320-CV-F2012-Rectification.pdf"""


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


# implement SIFT algorithm to find match points
def SIFT(src, dst):
    # SIFT
    sift = cv2.SIFT_create()

    kp1_SIFT, desc1_SIFT = sift.detectAndCompute(src, None)
    kp2_SIFT, desc2_SIFT = sift.detectAndCompute(dst, None)

    # which keypoints/descriptor to use?
    kp1 = kp1_SIFT
    kp2 = kp2_SIFT
    desc1 = desc1_SIFT
    desc2 = desc2_SIFT

    # (brute force) matching of descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    # Apply ratio test
    good_matches = []
    good_matches_without_list = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append([m])
            good_matches_without_list.append(m)
    # plt.imshow(img3),plt.show()
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches_without_list]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches_without_list]).reshape(-1, 1, 2)
    return src_pts, dst_pts


# Implement the eight-points algorithm by ourselves
def eight_point_alg(points1, points2):
    A = []
    for i in range(points2.shape[0]):
        A.append([points1[i][0]*points2[i][0], points1[i][1]*points2[i][0], points2[i][0], points2[i][1]*points1[i][0],
                  points1[i][1]*points2[i][1], points2[i][1], points1[i][0], points1[i][1], 1])
    A = np.asarray(A)

    # compute F_hat
    U,D,Vt = np.linalg.svd(A, full_matrices=True)
    D = list(D)
    F = Vt[D.index(min(D))].reshape(3, 3)

    # compute F
    Uf, Df, Vft = np.linalg.svd(F, full_matrices=True)
    Df = list(Df)
    Df[Df.index(min(Df))] = 0
    F = np.dot(Uf, np.dot(np.diag(Df), Vft))

    return F

# normalize the fundamental matrix
def normalized_F(p1, p2):
    # normalization
    points1_center_dis = p1 - np.average(p1, axis=0)
    points2_center_dis = p2 - np.average(p2, axis=0)

    # scale = 2 / mean square distance
    # compute two T matrix
    s1 = np.sqrt(2 / (np.sum(points1_center_dis ** 2) / p1.shape[0]))
    T_left = np.array([[s1, 0, -np.average(p1, axis=0)[0] ],
                   [0, s1, -np.average(p1, axis=0)[1] * s1],
                   [0, 0, 1]])
    s2 = np.sqrt(2 / (np.sum(points2_center_dis ** 2) / p1.shape[0]))
    T_right = np.array([[s2, 0, -np.average(p2[:, 0:2], axis=0)[0] * s2],
                   [0, s2, -np.average(p2[:, 0:2], axis=0)[0] * s2],
                   [0, 0, 1]])
    # compute the fundamental matrix
    Fq = eight_point_alg(np.transpose(np.dot(T_left, (np.transpose(p1)))),
                         np.transpose(np.dot(T_right, (np.transpose(p2)))))
    # de-normalize F
    F = np.dot(np.dot(np.transpose(T_right), Fq), T_left)

    return F


def iteration(F, src, dst, P):
    l = np.dot(np.transpose(F), np.transpose(dst))
    U, D, VT = np.linalg.svd(np.transpose(l))
    e = VT[-1] / VT[-1][2]
    EX = np.asarray([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
    Pprime = np.concatenate((EX, np.array([[e[0], e[1], e[2]]]).T), axis=1)
    x_hat = []
    x_hat_prime = []
    for i in range(8):
        A = np.array([src[i][0] * P[2].T - P[0].T,
                      src[i][1] * P[2].T - P[1].T,
                      dst[i][0] * Pprime[2].T - Pprime[0].T,
                      dst[i][1] * Pprime[2].T - Pprime[1].T])
        U, D, Vt = np.linalg.svd(A)
        D = list(D)
        Xi = Vt[D.index(min(D))]
        x_hat.append(np.dot(P, Xi))
        x_hat_prime.append(np.dot(Pprime, Xi))
    return np.asarray(x_hat), np.asarray(x_hat_prime), P, Pprime


def geometricDistance(F, src, dst, P):
    F = np.asarray([[F[0], F[1], F[2]], [F[3], F[4], F[5]],
                       [F[6], F[7], F[8]]])
    x_hat, x_hat_prime, P, Pprime = iteration(F, src, dst, P)
    sum = 0
    for i in range(8):
        sum += scipy.spatial.distance.euclidean(src[i], x_hat[i]) ** 2 + scipy.spatial.distance.euclidean\
            (x_hat[i], x_hat_prime[i].T)
    return sum


def MLEMethod(source, destination, intrinsic):
    F = normalized_F(source, destination)
    P = np.concatenate((intrinsic, np.array([[0, 0, 0]]).T), axis=1)
    F_mle = scipy.optimize.least_squares(geometricDistance, np.array([F[0][0], F[0][1], F[0][2], F[1][0],
                                                                      F[1][1], F[1][2], F[2][0], F[2][1], F[2][2]]),
                                         method="trf", args=(source, destination, P))
    return F_mle


def homographies(F, im2, points1, points2):
    # calculate epipole
    # l = F^T * p'
    l = np.dot(np.transpose(F), np.transpose(points2))
    U, D, VT = np.linalg.svd(np.transpose(l))
    e = VT[-1] / VT[-1][2]

    # calculate H2
    # first step
    # calculate th transform matrix
    width = im2.shape[1]
    height = im2.shape[0]
    T = np.asarray([[1, 0, -1.0 * width / 2], [0, 1, -1.0 * height / 2], [0, 0, 1]])

    # translated epipole Te'
    E = np.dot(T, e)
    if E[0] >= 0:
        alpha = 1
    else:
        alpha = -1
    # compute rotation matrix
    R = np.asarray([[alpha * E[0] / np.sqrt(E[0]**2 + E[1]**2), alpha * E[1] / np.sqrt(E[0]**2 + E[1]**2), 0],
                    [- alpha * E[1] / np.sqrt(E[0]**2 + E[1]**2), alpha * E[0] / np.sqrt(E[0]**2 + E[1]**2), 0],
                    [0, 0, 1]])
    # (f, 0, 0) is epipole line
    f = np.dot(R, E)[0]
    G = np.asarray([[1, 0, 0], [0, 1, 0], [- 1 / f, 0, 1]])
    H2 = np.dot(np.dot(np.dot(np.linalg.inv(T), G), R), T)

    # calculate H1
    # H1 = HA* H2 * M
    # first step: compute M
    # e_m is skew-symmetric
    e_m = np.asarray([[0, e[0], -e[1]], [-e[1], 0, e[2]], [e[1], -e[2], 0]])
    v = np.array([[1], [1], [1]])
    # M = [e] * F + e * v^T
    M = np.dot(e_m, (F)) + np.outer(e, np.transpose(v))

    # compute a value for HA
    ph1 = np.dot(np.dot(H2, M), np.transpose(points1))
    ph2 = np.dot(H2, np.transpose(points2))
    # least square problem Wa = b
    W = np.transpose(ph1)
    for i in range(W.shape[0]):
        W[i][0] /= W[i][2]
        W[i][1] /= W[i][2]
        W[i][2] /= W[i][2]
    b = np.transpose(ph2)[:, 0]

    # least square problem
    a = np.linalg.lstsq(W, b)[0]

    # Get HA
    HA = np.asarray([a, [0, 1, 0], [0, 0, 1]])
    H1 = np.dot(np.dot(HA, H2), M)
    return H1, H2


# contain two different normalize algorithm
def fundamental_Homo(choose, img1, img2, source, destination, intrinsics):
    # normalized eight point algorithm
    if choose == 1:
        F = normalized_F(source, destination)
        H1, H2 = homographies(F, img2, source, destination)
        return H1, H2
    else:
        F = MLEMethod(destination, source, intrinsics)
        F = np.asarray([[F.x[0], F.x[1], F.x[2]], [F.x[3], F.x[4], F.x[5]],
                         [F.x[6], F.x[7], F.x[8]]])
        H1, H2 = homographies(F, img2, source, destination)
        return H2, H2


# smooth left and right images
def denoise1(left,right,resize_shape,window_size,sigma):
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


def change_to_normal(past):
    new = []
    for i in range(past.shape[0]):
        if len(past[i][0]) == 3:
            x, y, z = past[i][0]
        else:
            y, x = past[i][0]
            z = 1
        new.append([x, y, z])
    return np.asarray(new)


def compute_depth_map(newImg1, newImg2):
    focal_length = 746.6666667
    T = abs(distance.euclidean((-1.529005, 1.516405, 1.636328), (-1.2999, 1.621381, 1.615433)))
    disparity_map, depth_map = Depthmap(newImg1, newImg2, focal_length, T, (5, 5), 10, 10)
    return disparity_map, depth_map


# main function
def general2parallel(img1, img2):
    # the data from the data set
    source, destination = SIFT(img1, img2)
    intrinsics = np.loadtxt('../Test_images/teapot1/intrinsics.txt')
    mp1 = change_to_normal(source)
    mp2 = change_to_normal(destination)
    H1, H2 = fundamental_Homo(1, img1, img2, mp1, mp2, intrinsics)
    H1_l, H2_l = fundamental_Homo(2, img1, img2, mp1, mp2, intrinsics)
    size = (img1.shape[0], img1.shape[0])
    im1 = cv2.imread('../Test_images/third_pipeline/0.jpg')
    im2 = cv2.imread('../Test_images/third_pipeline/1.jpg')
    newImg1 = cv2.rotate(cv2.warpPerspective(im1, H1, size), cv2.ROTATE_90_CLOCKWISE)
    newImg2 = cv2.rotate(cv2.warpPerspective(im2, H2, size), cv2.ROTATE_90_CLOCKWISE)
    plt.title("Left parallel image for first normalized algorithm")
    plt.imshow(newImg1)
    plt.show()
    plt.title("Right parallel image for first normalized algorithm")
    plt.imshow(newImg2)
    plt.show()
    newImg1_l = cv2.rotate(cv2.warpPerspective(im1, H1_l, size), cv2.ROTATE_90_CLOCKWISE)
    newImg2_l = cv2.rotate(cv2.warpPerspective(im2, H2_l, size), cv2.ROTATE_90_CLOCKWISE)
    plt.title("Left parallel image for second normalized algorithm")
    plt.imshow(newImg1_l)
    plt.show()
    plt.title("Right parallel image for second normalized algorithm")
    plt.imshow(newImg2_l)
    plt.show()
    left_blur, right_blur = denoise(cv2.rotate(cv2.warpPerspective(img1, H1, size), cv2.ROTATE_90_CLOCKWISE),
                                    cv2.rotate(cv2.warpPerspective(img2, H2, size), cv2.ROTATE_90_CLOCKWISE),
                                    (int(img1.shape[1] // 2.3), int(img1.shape[0] // 2.3)), (5, 5), 1)
    disparity, depth_map = compute_depth_map(left_blur, right_blur)
    # test_error(depth_map)
    plt.title("disparity map")
    plt.imshow(disparity)
    plt.show()
    plt.title("depth map")
    plt.imshow(depth_map)
    plt.show()
    newImg1 = cv2.resize(newImg1, (img1.shape[1] // 2.3, img1.shape[0] // 2.3))
    #pc = compute_point_cloud(depth_map, newImg1, extrinsics, intrinsics)
    #np.savetxt('../Final_results/Third_pipeline/pointcloud.txt', pc)
    #plot_pointCloud(pc)


im1 = cv2.imread('../Test_images/third_pipeline/0.jpg', cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread('../Test_images/third_pipeline/1.jpg', cv2.IMREAD_GRAYSCALE)

general2parallel(im1, im2)
