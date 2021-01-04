from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
import numpy as np
import time
import open3d as o3d
import plotly.graph_objects as go

def calculate_centriod(pc):
    '''Return the centroids of a pointcloud in the order of x,y,z '''
    x_mean = np.mean(pc[0])
    y_mean = np.mean(pc[1])
    z_mean = np.mean(pc[2])
    centroids = np.array([[x_mean],[y_mean],[z_mean]])
    return centroids

def compute_transform(pc1,pc2):
    '''Compute the transformation(R and T) from pc1(source) to pc2 (reference fixed).
    pc1:  3xnum_points
    pc2: 3xnum_points'''
    #Find the centroids of 2 point clouds.
    centroid1 = calculate_centriod(pc1)
    centroid2 = calculate_centriod(pc2)

    #accumulating a matrix H
    pc1_centered = pc1 - centroid1 #shape 3xnum_points
    pc2_centered = pc2 - centroid2
    H = pc1_centered@np.transpose(pc2_centered) #shape: 3x3
    #Find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # special reflection case
    if np.linalg.det(R) < 0:
        print("reflection occurs")
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    #Translation matrix
    T = centroid2-R@centroid1 #shape:3x1
    return R, T

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nx3 array of points
        dst: Nx3 array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''
    #Borrowed method in stack-overflow
    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def KDTree_matching(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst(fixed point cloud) for each point in src
    Input:
        src: Nx3 array of points
        dst: Nx3 array of points(fixed point cloud)
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''
    # Find the nearest points for each point in src among the points in dst
    kdtree = KDTree(dst)
    dist, indices = kdtree.query(src, 1)
    #print('dist shape is,',dist.shape) #dist shape is, (110592,)
    #print('indices shape is,', indices.shape)# indices shape is, (110592,)
    return dist, indices



def icp(pc1, pc2, threshold, num_iter, KP_Tree):
    """Return the transformed source point cloud.
    :param pc1: source point cloud(3xN)
    :param pc2: reference point cloud(fixed)
    :param threshold: Transfomation is the best when SSD less than threshold,
    :param num_iter: maximum number of iteration(in case too time consuming)
    :return: Best transformation from pc1 to pc2
    """
    num_pts = pc1.shape[1]
    iter = 0
    source_pc = pc1

    while iter<=num_iter:
        print('Number of iterations is,',iter)
        # find the nearest neighbors between the current source and destination points
        # choose which algorithm to match closest point
        if KP_Tree:
            dist, indices = KDTree_matching(source_pc.T, pc2.T)
        else:
            dist, indices = nearest_neighbor(source_pc.T, pc2.T)
        #find transformation which will best align each source point to its match found
        R, T = compute_transform(source_pc[:,indices.ravel()],pc2)
        new_pc = (R@ source_pc)+T
        # Find the root mean squared error(variant of Sum of squared difference(SSD))
        error = np.square(dist)
        SSD = np.sum(error)
        root_mSSD= np.sqrt(SSD/num_pts)
        print('Error is,',root_mSSD)
        if root_mSSD < threshold:
            break
        else:
            source_pc = new_pc
            #generate transformation for the next iteration
            R,T = compute_transform(source_pc,pc2)
            #print("Rotation matrix is,", R)
            #print("Translation matrix is,", T)
        iter+=1
    #transformed source pc
    transformed_pc = source_pc
    return R,T, transformed_pc

def Render_merged_pc(transformed_pc, pc1,pc2):
    """
    Return the merged point cloud(2Nx6) given the best transformation found by ICP.
    :param transformed_pc: source point cloud(3xN)
    :param pc1: source point cloud (Nx6)
    :param pc2: reference point cloud(fixed)(Nx6)
    """
    #Switch the xyz column of pc1 to transformed version
    pc1[:,:3] = transformed_pc.T
    merged_pc  = np.append(pc1,pc2,axis=0)
    return merged_pc

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

    pc1 = np.loadtxt("../Images/teapot1/pointcloud.txt")#shape Nx6
    pc2 = np.loadtxt("../Images/teapot2/pointcloud.txt")#shape Nx6
    #print(pc1.shape)
    #print(pc2.shape)
    # find the xyz coordinates of source point cloud and reference point cloud
    pc1_coor = (pc1[:, :3]).T# shape:3xN
    pc2_coor = (pc2[:, :3]).T  # shape:3xN
    timer1 = time.perf_counter()
    R, T, transformed_pc = icp(pc1_coor, pc2_coor, 0.0001, num_iter=100, KP_Tree=False)
    timer2 = time.perf_counter()
    print('The overall algorithm takes ' + str(timer2-timer1) + 'seconds.')
    #np.savetxt('../Result/First_pipeline/pointcloud_icp.txt', transformed_pc.T)
    merged_pc = Render_merged_pc(transformed_pc, pc1, pc2)
    np.savetxt('../Final_results/First_pipeline/pointcloud_icp_nn.txt', merged_pc)
    plot_pointCloud(merged_pc)