import numpy as np
import os
import matplotlib.pyplot as plt
from util.camera_pose_visualizer import CameraPoseVisualizer

filepath = 'C:\image\kitti2\poses.txt'
new_path = 'C:\image\poses.txt' 

def load_poses(filepath):
        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The GT poses
        """
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses


poses = np.array(load_poses(new_path))

print(poses[0])

visualizer = CameraPoseVisualizer([-10, 10], [-10, 10], [0, 10])
i =0
for pose in poses:
    # if i%3 == 0:
    #     visualizer.extrinsic2pyramid(pose, 'c', 5)
    # i+=1
    visualizer.extrinsic2pyramid(pose, 'c', 0.5)

visualizer.show()