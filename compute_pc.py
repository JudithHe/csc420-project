import numpy as np
import matplotlib.pyplot as plt
import cv2
import plotly.graph_objects as go
import open3d as o3d

#All codes are from starter code for A4 and my own implementation
def modify_intrinsics(intrinsics,shape):
  '''Return the modified intrinsics matrix in case that the image is resized
  Need to update px,py'''
  rows,cols = shape
  #print(shape)
  new_px = int(cols / 2)
  new_py = int(rows / 2)
  intrinsics[0,2] = new_px
  intrinsics[1,2] = new_py
  return intrinsics


def get_data(folder_name):
  '''
  reads data in the specified image folder
  '''

  rgb = cv2.imread(folder_name + 'rgbImage.png')
  rows,cols = rgb.shape[0:2]
  depth_nyu = cv2.imread(folder_name + 'depth_nyu.png')[:,:,0]
  #depth_kitti = cv2.imread(folder + 'depth_kitti.png')[:, :, 0]
  extrinsics = np.loadtxt(folder_name + 'extrinsics.txt')
  intrinsics = np.loadtxt(folder_name + 'intrinsics.txt')
  intrinsics = modify_intrinsics(intrinsics, (rows,cols))
  return depth_nyu, rgb, extrinsics, intrinsics


def compute_point_cloud(imageNumber):
  '''
  Return a N x 6 matrix where N is the number of 3D points with 3
  coordinates and 3 color channel values given depth map, rgb,
  extrinsics and intrinsics of a 2D image.
  '''
  depth, rgb, extrinsics, intrinsics = get_data(imageNumber)
  #Extract RBG channels of rgb image each has shape (rows,cols)
  Red = rgb[:,:,0]
  Green = rgb[:,:,1]
  Blue = rgb[:,:,2]

  rows,cols = depth.shape
  N = rows*cols
  print('number of points is,',rows,cols)
  results = np.zeros((N,6))
  n=0

  for i in range(rows):
    for j in range(cols):
      #pixel coordinates of 2D image plane
      #we switch x and y since origin of image plane is at bottom right
      d = depth[i,j]
      x = j*d
      y = (rows-1-i)*d
      left = np.array([[x],[y],[d]])
      #Find the camera coordinates: check a = Zc(yes)
      camera = np.linalg.inv(intrinsics)@left #shape is (3,1)
      homo_camera = np.array([[camera[0,0]],[camera[1,0]],[camera[2,0]],[1]])
      #Find the world coordinates
      #extrinsics is homogenous extrinsics(4x4)
      world_coord = np.linalg.inv(extrinsics)@homo_camera
      #store in results list
      results[n,0] = world_coord[0,0]
      results[n,1] = world_coord[1,0]
      results[n,2] = -world_coord[2,0]
      #store color channel values
      results[n,3] = Red[i,j]
      results[n,4] = Green[i,j]
      results[n,5] = Blue[i,j]
      n = n+1
  return results

def plot_pointCloud(pc):
  '''
  plots the Nx6 point cloud pc in 3D
  assumes (1,0,0), (0,1,0), (0,0,-1) as basis
  '''
  fig = go.Figure(data=[go.Scatter3d(
      x=pc[:, 0],
      y=pc[:, 1],
      z=pc[:, 2],
      mode='markers',
      marker=dict(
          size=2,
          color=pc[:, 3:][..., ::-1],
          opacity=0.8
      )
  )])
  fig.show()


if __name__ == '__main__':
  #rgb = cv2.imread('./Images/teapot1/rgbImage.png')
  #plt.imshow(rgb)
  #plt.show()
  folders = ['./Images/teapot2/']
  for foldername in folders:
    #depth, rgb, extrinsics, intrinsics = get_data(foldername)
    #print(intrinsics)
    #point_cloud=depth_to_voxel_ld(depth, scale=3)
    pc = compute_point_cloud(foldername)
    np.savetxt(foldername + 'pointcloud.txt', pc)
    plot_pointCloud(pc)
