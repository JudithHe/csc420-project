import numpy as np
import time
import open3d as o3d
import plotly.graph_objects as go

def rigidTransformation(A, B):
    """
    Match two point clouds using least square lost.

    A' = R*A.T + t, where A' should be close to B.T

    Shape of the rigid bodies will not change after transformation.
    :param A: point cloud A (shape: Nx3)
    :param B: point cloud B (shape: Nx3)
    :return: Return R and t for rigid transformation
    """
    # nx3 -> 3xn
    A = A.T
    B = B.T

    # center and rotate
    center_A = np.mean(A, axis=1).reshape(-1, 1)
    center_B = np.mean(B, axis=1).reshape(-1, 1)
    H = np.matmul((A - center_A), (B - center_B).T)

    U, S, Vt = np.linalg.svd(H)
    Rotation = np.matmul(Vt.T, U.T)

    if np.linalg.det(Rotation) < 0:
        Ur, Sr, Vtr = np.linalg.svd(Rotation)
        Vtr[2, :] *= -1
        Rotation = np.matmul(Vtr.T, Ur.T)

    return Rotation, center_B - np.matmul(Rotation, center_A)


def computeAPrimeT(A, r, t):
    """
    Given r and t for rigid body transformation and compute the transpose of A' (A' = R*A.T + t)
    :param A: point cloud A (shape:Nx3)
    :param r: rotation
    :param t: translation
    :return: transpose of A', which should be close to B (shape Nx3)
    """
    return (np.matmul(r, A.T) + t).T


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

if __name__ == "__main__":
    timer = time.perf_counter()

    A = np.loadtxt("../Images/teapot1/pointcloud.txt")  # shape Nx6
    B = np.loadtxt("../Images/teapot2/pointcloud.txt")  # shape Nx6
    A_coor = (A[:, :3])  # shape:Nx3
    B_coor = (B[:, :3])  # shape:Nx3
    r, t = rigidTransformation(A_coor, B_coor)
    transformed_A = computeAPrimeT(A_coor, r, t) #shape Nx3
    merged_pc = Render_merged_pc(transformed_A.T, A, B)
    np.savetxt('../Final_result/First_pipeline/pointcloud_rigid_rendered.txt', merged_pc)

    print(time.perf_counter() - timer)

    plot_pointCloud(merged_pc)