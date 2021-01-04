import argparse
from ICP import *
from rigid3D import *

parser = argparse.ArgumentParser()
parser.add_argument('method', metavar='N', type=str, nargs='+')

args = parser.parse_args()

if args.method[0].lower() == "icp":
    pc1 = np.loadtxt("../Test_Images/teapot1/pointcloud.txt")#shape Nx6
    pc2 = np.loadtxt("../Test_Images/teapot2/pointcloud.txt")#shape Nx6
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
elif args.method[0].lower() == "rigid":
    timer = time.perf_counter()

    A = np.loadtxt("../Test_Images/teapot1/pointcloud.txt")  # shape Nx6
    B = np.loadtxt("../Test_Images/teapot2/pointcloud.txt")  # shape Nx6
    A_coor = (A[:, :3])  # shape:Nx3
    B_coor = (B[:, :3])  # shape:Nx3
    r, t = rigidTransformation(A_coor, B_coor)
    transformed_A = computeAPrimeT(A_coor, r, t) #shape Nx3
    merged_pc = Render_merged_pc(transformed_A.T, A, B)
    np.savetxt('../Final_results/First_pipeline/pointcloud_rigid_rendered.txt', merged_pc)

    print(time.perf_counter() - timer)

    plot_pointCloud(merged_pc)