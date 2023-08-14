import numpy as np
from util.camera_pose_visualizer import CameraPoseVisualizer

if __name__ == '__main__':
    # argument : the minimum/maximum value of x, y, z
    visualizer = CameraPoseVisualizer([-50, 50], [-50, 50], [0, 50])

    old_pose = np.array([[ 0.9999, 0.0067, 0.0097, -4.6075],
                        [ -0.0067, 1.0000, 0.0056, -0.7015],
                        [ -0.5, -0.0056, 0.9999, 0.1200],
                        [ 0, 0, 0, 1]]) #first pose T
    new_pose = np.array([[ 1.000000e+00, 1.197625e-11, 1.704638e-10, 0.000000e+00],
                            [ 1.197625e-11, 1.000000e+00, 3.562503e-10, -1.110223e-16],
                            [ 1.704638e-10, 3.562503e-10, 1.000000e+00, 2.220446e-16],
                            [ 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]]) #kitti1
    
    # argument : extrinsic matrix, color, scaled focal length(z-axis length of frame body of camera
    # 
    visualizer.extrinsic2pyramid(old_pose, 'c', 10)
    #visualizer.extrinsic2pyramid(np.eye(4), 'c', 10)
    #visualizer.extrinsic2pyramid(new_pose, 'c', 10)

    visualizer.show()
