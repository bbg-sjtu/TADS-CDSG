import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from SE3_DS.utils.utils import *
from sophus_pybind import SE3
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader, Dataset
import cv2
class DIFFIEO_DATASET(Dataset):
    def __init__(self, trajs, scaling_factors, device, PLOT=False):
        self.traj=trajs
        self.scaling_factors=scaling_factors
        self.traj_num=1
        self.dim=self.traj.shape[1]
        self.traj_len=self.traj.shape[0]
        self.target=torch.zeros((self.traj_len,self.dim),dtype=torch.float32)
        weights = torch.linspace(1, 0, self.traj_len)
        for i in range(self.dim):
            self.target[:,i]=weights
        self.traj=torch.tensor(self.traj,dtype=torch.float32)
        x_end=self.traj[-1,:].unsqueeze(0).repeat(500, 1)
        y_end=torch.zeros((500,self.dim),dtype=torch.float32)
        self.target=torch.cat((self.target,y_end),dim=0)
        self.traj=torch.cat((self.traj,x_end),dim=0)
        self.traj_len=self.traj.shape[0]

        self.traj=self.traj.to(device)
        self.target=self.target.to(device)

        # # plot
        if PLOT==True:
            fig = plt.figure()
            for i in range(3):
                plt.plot(self.traj[:,0].cpu())
                plt.plot(self.traj[:,1].cpu())
                plt.plot(self.traj[:,2].cpu())
                plt.plot(self.target[:,0].cpu())

            plt.show()
            print('yes')

    def __len__(self):
        return self.traj_len

    def __getitem__(self, index):
        X=self.traj[index,:]
        X_tatget=self.target[index,:]
        return X, X_tatget

class SE3_DATASET():
    def __init__(self, data_path, k=1500, device = torch.device('cuda'), PLOT=False):
        self.type = type
        self.dim = 6
        self.dt = .01
        traj_num=1
        self.trajs_real=[]
        for index in range(traj_num):
            self.trajs_real.append(np.loadtxt(data_path, delimiter=','))

        self.n_trajs = len(self.trajs_real)
        self.n_dims = self.trajs_real[0].shape[-1]

        self.rot_trajs = []

        for trj in self.trajs_real:
            trj_rot = self.xyz_rotvec_2_matrix(trj[0:k,:])
            self.rot_trajs.append(trj_rot)

        self.init_vec = []
        for i in range(len(self.rot_trajs)):
            self.init_vec.append(self.rot_trajs[i][0,:])

        if traj_num==1:
            goal_H=self.rot_trajs[-1][-1,:,:]
        else:
            goal_H = self.compute_se3_center(self.rot_trajs)
        self.goal_H = goal_H
        self.norm_rot_trajs = []
        for trj in self.rot_trajs:
            n_trj = np.zeros((0,4,4))
            for t in range(trj.shape[0]):
                H = trj[t,...]
                H2 = np.matmul(np.linalg.inv(H), goal_H)
                n_trj = np.concatenate((n_trj, H2[None,...]),0)
            self.norm_rot_trajs.append(n_trj)

        if PLOT:
            fig = plt.figure(figsize=(20, 20), num=1)
            ax = fig.add_subplot(111, projection='3d')
            for i in range(traj_num):
                visualize_SE3_traj(self.norm_rot_trajs[i][:,:], ax=ax, traj_color='blue')
                visualize_SE3_traj(self.norm_rot_trajs[i][:1200,:], ax=ax, traj_color='red')
            plt.show()

        self.se3_trajs = []
        for trj in self.norm_rot_trajs:
            trj_ax_angle = self.SE3_rot_traj_2_se3_traj(trj)
            self.se3_trajs.append(trj_ax_angle)

        if PLOT:
            fig, axs = plt.subplots(6)
            fig.suptitle('Vertically stacked subplots')
            for trj in self.se3_trajs:
                for i in range(6):
                   axs[i].plot(trj[:,i])
            plt.show()
        print('yes')

        # normalize data
        self.normalized_se3_trajs = np.zeros_like(self.se3_trajs[0], dtype=np.float64)
        self.scaling_factors = np.zeros(6, dtype=np.float64)
        for i in range(6):
            dim_data = self.se3_trajs[0][:, i]
            abs_data = np.abs(dim_data)
            if np.max(abs_data) == 0:
                scaling_factor = 1.0
            else:
                scaling_factor = np.max(abs_data)

            self.normalized_se3_trajs[:, i] = dim_data / scaling_factor
            self.scaling_factors[i] = scaling_factor

        self.dataset = DIFFIEO_DATASET(trajs=self.normalized_se3_trajs, scaling_factors=self.scaling_factors, device=device, PLOT=PLOT)

    def compute_se3_center(self, rot_trajectories):
        goals = np.zeros((0,4,4))
        for trj in rot_trajectories:
            goals = np.concatenate((goals,trj[-1:,...]),0)

        mean = np.eye(4)
        opt_steps = 10
        for i in range(opt_steps):
            axis_array = np.zeros((0,6))
            for t in range(goals.shape[0]):
                g = goals[t, ...]
                g_I = np.matmul(np.linalg.inv(mean), g)
                axis = SE3.from_matrix(g_I).log()
                axis_array = np.concatenate((axis_array,axis),0)

            mean_tangent = np.mean(axis_array,0)
            mean_new = SE3.exp(mean_tangent[:3], mean_tangent[3:]).to_matrix()
            mean = np.matmul(mean, mean_new)
        return mean

    def SE3_rot_traj_2_se3_traj(self, rot_traj):
        ax_angle_trj = np.zeros((0, 6))
        for t in range(rot_traj.shape[0]):
            vt = SE3.from_matrix(rot_traj[t, ...]).log()
            ax_angle_trj = np.concatenate((ax_angle_trj, vt), 0)
        return ax_angle_trj

    def xyz_rotvec_2_matrix(self, trj):

        matrix_trj = np.zeros((trj.shape[0], 4, 4))
        
        for i in range(trj.shape[0]):
            translation = trj[i, :3]
            rot_vec = trj[i, 3:]
            angle = np.linalg.norm(rot_vec)
            if angle < 1e-8:
                rotation_matrix = np.eye(3)
            else:
                rotation = R.from_rotvec(rot_vec)
                rotation_matrix = rotation.as_matrix()
            H = np.eye(4)
            H[:3, :3] = rotation_matrix
            H[:3, 3] = translation
            matrix_trj[i] = H
        return matrix_trj


if __name__ == "__main__":
    from pathlib import Path
    _THIS_DIR = Path(__file__).resolve().parent
    DATA_DIR = _THIS_DIR / "PegInHole_data/teach_data_000.txt"
    data=SE3_DATASET(data_path=DATA_DIR, k=1085, PLOT=True)
    print('yes')
