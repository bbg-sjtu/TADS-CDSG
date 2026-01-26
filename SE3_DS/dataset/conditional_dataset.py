import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from SE3_DS.utils.utils import *
from sophus_pybind import SE3
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader, Dataset
import cv2

class CONDITIONAL_DIFFIEO_DATASET(Dataset):
    def __init__(self, trajs, device, traj_num):
        data_len=1500

        # Conditional Data
        traj_startidx=np.array([0,0,0,0])
        traj_endidx=np.array([1120,1110,1310,1350])
        self.condition=np.array([0.25,0.5,0.75,1.0])

        traj_len=traj_endidx-traj_startidx

        self.trajs=torch.zeros((traj_num*data_len,7), dtype=torch.float32)
        self.target=torch.zeros((traj_num*data_len,6), dtype=torch.float32)

        for i in range(traj_num):
            condition=self.condition[i]*torch.ones((data_len), dtype=torch.float32)
            traj=torch.tensor(trajs[i, traj_startidx[i]:traj_endidx[i], :], dtype=torch.float32)
            traj_end=traj[-1,:].unsqueeze(0).repeat(data_len-traj.shape[0], 1)
            traj=torch.cat((traj, traj_end), dim=0)
            self.trajs[i*data_len:(i+1)*data_len,:6]=traj
            self.trajs[i*data_len:(i+1)*data_len,6]=condition

            weights = torch.linspace(1, 0, traj_len[i])
            end=torch.zeros((data_len-traj_len[i]), dtype=torch.float32)
            target=torch.cat((weights,end),dim=0)
            self.target[i*data_len:(i+1)*data_len,:]=target.unsqueeze(1)

        self.traj_len=self.trajs.shape[0]
        self.trajs=self.trajs.to(device)
        self.target=self.target.to(device)

    def __len__(self):
        return self.traj_len

    def __getitem__(self, index):
        xyzc=self.trajs[index,:]
        X=xyzc[:6]
        condition=xyzc[6].unsqueeze(0)
        tatget=self.target[index,:]
        return X, condition, tatget

class CONDITIONAL_IMG_DIFFIEO_DATASET(Dataset):
    def __init__(self, trajs, imgs, device, traj_num):
        data_len=1500

        # # Conditional Data
        traj_startidx=np.array([0,0,0,0])
        traj_endidx=np.array([1400,1350,1500,1400])

        traj_len=traj_endidx-traj_startidx

        self.trajs=torch.zeros((traj_num*data_len,6), dtype=torch.float32)
        self.imgs = torch.zeros((traj_num*data_len,3,224,224), dtype=torch.float32)
        self.target=torch.zeros((traj_num*data_len,6), dtype=torch.float32)

        for i in range(traj_num):
            traj=torch.tensor(trajs[i,traj_startidx[i]:traj_endidx[i],:], dtype=torch.float32)
            traj_end=traj[-1,:].unsqueeze(0).repeat(data_len-traj.shape[0], 1)
            traj=torch.cat((traj, traj_end), dim=0)
            self.trajs[i*data_len:(i+1)*data_len,:]=traj
            weights = torch.linspace(1, 0, traj_len[i])
            end=torch.zeros((data_len-traj_len[i]),dtype=torch.float32)
            target=torch.cat((weights,end),dim=0)
            self.target[i*data_len:(i+1)*data_len,:]=target.unsqueeze(1)
            # images
            images=torch.tensor(imgs[i,traj_startidx[i]:traj_endidx[i],:,:,:], dtype=torch.float32)
            images_end=images[-1,:,:,:].unsqueeze(0).repeat(data_len-traj_len[i],1,1,1)
            result = torch.cat([images, images_end], axis=0)
            self.imgs[i*data_len:(i+1)*data_len,:,:,:] = result

        self.traj_len=self.trajs.shape[0]
        self.trajs=self.trajs.to(device)
        self.target=self.target.to(device)
        self.imgs=self.imgs.to(device)

    def __len__(self):
        return self.traj_len

    def __getitem__(self, index):
        X=self.trajs[index,:]
        condition_img=self.imgs[index,:,:,:]
        target=self.target[index,:]
        return X, condition_img, target


class CONDITIONAL_DATASET():
    def __init__(self, data_path, device = torch.device('cuda'), PLOT=False, isImg=True):

        traj_num=4
        self.trajs_real=[]
        for index in range(traj_num):
            self.trajs_real.append(np.loadtxt(f"{data_path}/teach_data_{index:03d}.txt", delimiter=','))

        self.rot_trajs = []
        k=1500
        print('Loading trajectories...')
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
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111, projection='3d')
            for i in range(traj_num):
                visualize_SE3_traj(self.rot_trajs[i][:,:], ax=ax1, traj_color='blue')

            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111, projection='3d')
            for i in range(traj_num):
                visualize_SE3_traj(self.norm_rot_trajs[i][:,:], ax=ax2, traj_color='blue')
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
            
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111, projection='3d')
            matrix_recon=self.se3_traj_2_SE3_rot_traj_task(np.array(self.se3_trajs).reshape(-1, 6), goal_H)
            for i in range(4):
                visualize_SE3_traj(matrix_recon[i*1500:(i+1)*1500,:], ax=ax1, traj_color='blue')
            plt.show()
        print('yes')


        # normalize
        self.se3_trajs=np.array(self.se3_trajs)
        self.normalized_se3_trajs = np.zeros_like(self.se3_trajs, dtype=np.float64)
        self.scaling_factors = np.zeros(6, dtype=np.float64) # different trajs share the same scaling factor！

        for j in range(6):
            dim_data = self.se3_trajs[: , :, j]
            abs_data = np.abs(dim_data)
            if np.max(abs_data) == 0:
                scaling_factor = 1.0
            else:
                scaling_factor = np.max(abs_data)

            self.normalized_se3_trajs[: , :, j] = dim_data / scaling_factor
            self.scaling_factors[j] = scaling_factor

        if isImg:
            # load images
            print('Loading images...')
            self.imgs=np.zeros((traj_num,k,224,224,3), dtype=np.float32)
            for index in range(traj_num):
                img_path=f"{data_path}//teach_data_{index:03d}_imgs"
                for i in range(k):
                    image_filename = f"{i+1:06d}.jpg"
                    image_path = os.path.join(img_path, image_filename)
                    img = cv2.imread(image_path)
                    img_resized = cv2.resize(img, (224, 224))
                    self.imgs[index,i,:,:,:]=(img_resized/ 255.0 - 0.5) / 0.5
            self.imgs = np.transpose(self.imgs, (0, 1, 4, 2, 3))

            self.dataset = CONDITIONAL_IMG_DIFFIEO_DATASET(trajs=self.normalized_se3_trajs, imgs=self.imgs, device=device, traj_num=traj_num)
        else:
            self.dataset = CONDITIONAL_DIFFIEO_DATASET(trajs=self.normalized_se3_trajs, device=device, traj_num=traj_num)

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

    def se3_traj_2_SE3_rot_traj_task(self, se3_traj, goal_H):
        rot_traj = np.zeros((0, 4, 4))
        for t in range(se3_traj.shape[0]):
            H = SE3.exp(se3_traj[t, :3], se3_traj[t, 3:]).to_matrix()
            H = np.matmul(goal_H, np.linalg.inv(H))
            rot_traj =  np.concatenate((rot_traj, H[None, ...]), 0)
        return rot_traj
