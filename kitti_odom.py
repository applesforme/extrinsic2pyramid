import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from util.camera_pose_visualizer import CameraPoseVisualizer

pictures=[]
img_path=[]

path = r'C:\image\kitti1\image_l'
filepath = 'C:\image\kitti1'

for pictures in os.listdir(path):
    img_path.append(os.path.join(path, pictures))


class imgprocess(object):
    def __init__(self):
        #Extracting features using SIFT algorithms
        self.sift = cv.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        self.flann = cv.FlannBasedMatcher(index_params,search_params)
        
        self.last = None
        self.old_pose = None
        self.new_pose = None
        self.gt_poses = self.load_poses(os.path.join(filepath, 'poses.txt'))
        self.estimated_poses = []

        #Camera's instrinsic mat 
        self.K = self.load_calib(os.path.join(filepath, 'calib.txt'))

    def load_poses(self, filepath):
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
    
    def load_calib(self, filepath):
        """
        Loads the intrinsic camera matrix

        Parameters
        ----------
        filepath (str): The file path to the calib file

        Returns
        -------
        K (3*3 ndarray): Camera intrinsic matrix
        """
        with open(filepath, 'r') as f:
            for line in f.readlines():
                params = np.fromstring(line, dtype=np.float64, sep=' ')
                P = np.reshape(params, (3,4))
                K = P[0:3, 0:3]
        return K

    def extractandmatch(self, img_array):
        gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY) 

        #Detection (detectAndCompute function created noises)
        feats = cv.goodFeaturesToTrack(gray, maxCorners = 3000, qualityLevel = 0.01, minDistance = 3)
        
        #Feats -> keypoints
        kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats] 
        kps, des = self.sift.compute(gray, kps)
        
        matches = None
        pts1 = []
        pts2 = []
        
        #Matching
        if self.last is not None:
            matches = self.flann.knnMatch(des, self.last['des'], k=2)
            
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                    pts1.append(kps[m.queryIdx].pt)
                    pts2.append(self.last['kps'][m.trainIdx].pt)

        self.last = {'kps': kps, 'des': des}        
        

        #To find fundamental(essential) matrices
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        data = np.stack((pts1, pts2), axis=1)
    
        return data
    
    #Getting visual odometry
    def getcameraposes(self, data):
        if len(data)>0:
            #find essential mat and recover Rotation and translation
            E, _ = cv.findEssentialMat(data[:,0], data[:,1], self.K, threshold=1)
            retval, R, t, _ = cv.recoverPose(E, data[:,0], data[:,1], self.K, 1)

            #transformation matrix
            T = np.eye(4, dtype=np.float64) 
            T[:3, :3] = R # 3 rows and colums
            T[:3, 3] = t.T # 3rd column

            # Saving the rotated and translated points new_pose = old_pose * T
            if R is not None and t is not None:
                if self.old_pose is None:
                    self.old_pose = np.array(self.gt_poses[0])                   
                    self.estimated_poses.append([self.old_pose[0,3],self.old_pose[1,3], self.old_pose[2,3]])
        
                else:
                    self.old_pose = self.new_pose.copy()

                self.new_pose = np.matmul(T, self.old_pose)
                self.estimated_poses.append([self.new_pose[0,3], self.new_pose[1,3], self.new_pose[2,3]])    
                
        return self.estimated_poses
    
    def returnGT(self):
        return self.gt_poses


    #Visualization of camera poses
    # def visualize(self, estimated_poses):
    #     fig = plt.figure(figsize=(10,10))
    #     ax = fig.add_subplot(111, projection='3d')

    #     for points in estimated_poses:
    #         ax.scatter3D(points[0], points[2], points[1], c='b', marker='x', label='Estimation')
    #         #plt.pause(0.5)

    #     for points in self.gt_poses:
    #         ax.scatter3D(points[0,3], points[2,3], points[1,3], c='g', marker='o', label='Ground truth')
    #         ax.set_xlabel('X')
    #         ax.set_ylabel('Y')
    #         ax.set_zlabel('Z')

    #     #ax.legend(), to properly add a legend you should not use for statement with ax.scatter3D
    #     plt.axis('equal')
    #     plt.show()


ip = imgprocess()


for pictures in img_path:
    img_array = cv.imread(pictures)
    ret = ip.extractandmatch(img_array)
    estimated_poses =ip.getcameraposes(ret)
    
    for p1, p2 in ret:
        u0,v0 = map(lambda x: int(round(x)), p1)
        cv.circle(img_array, (u0, v0), radius=1, color=(0, 255, 0), thickness=1)
        u1,v1 = map(lambda x: int(round(x)), p2)
        cv.circle(img_array, (u1, v1), radius=1, color=(0, 255, 0), thickness=1)
        cv.line(img_array, (u0,v0), (u1,v1), color=(255, 0,0), thickness=1)



    cv.imshow('img_array', img_array)
    if cv.waitKey(1) == 27:
        break

cv.destroyWindow('img_array')

visualizer = CameraPoseVisualizer([-50, 50], [-50, 50], [0, 50])

gt_poses = np.array(ip.returnGT())

print(gt_poses)
#visualizer.extrinsic2pyramid(gt_poses, 'c', 10)

#visualizer.show()



# to triangulate points, you should get inlier points first, then you solve the equation...
# it's monocular, so it must be transformed to match the first frame