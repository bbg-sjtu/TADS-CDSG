import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.utils import *
from sophus_pybind import SE3
import numpy as np

device = 'cuda'  if torch.cuda.is_available() else 'cpu'
class Trainer:
    def __init__(self, model, dataset,batch_size=64, num_epochs=500):
        self.model = model.to(device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs


    def train(self, save_path):
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(),lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.8)
        best_train_loss = float('inf')
        for epoch in range(self.num_epochs):
            total_loss1=0
            total_loss2 = 0
            total_loss3 = 0
            total_loss = 0
            for _, (x, y_target) in enumerate(self.dataloader):
                if x.dtype != torch.float32:
                    x = x.to(torch.float32)
                    y_target = y_target.to(torch.float32)
                
                y_pred, J = self.model(x)
                loss_forward = F.mse_loss(y_pred, y_target)
                
                x_recon, _ = self.model(y_target,'inverse')
                loss_backward = F.mse_loss(x_recon, x)
                
                loss_jacobian = 0.0005*self.jacobian_regularization(J)
                loss1=loss_forward
                loss2=loss_backward
                loss3=loss_jacobian
                loss = loss_forward + loss_backward + loss_jacobian
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            total_loss3 += loss3.item()
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {total_loss}, Loss1: {total_loss1}, Loss2: {total_loss2}, Loss3: {total_loss3}, LR: {scheduler.get_last_lr()[0]}')
            
            if total_loss < best_train_loss:
                best_train_loss = total_loss
                torch.save(self.model.state_dict(), save_path)
                print(f'Saved best model to {save_path}')

    def test(self, goal_H, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.dataloader = DataLoader(self.dataset, batch_size=self.dataset.traj_len, shuffle=False)
        x, y_target = next(iter(self.dataloader))
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
            y_target = y_target.to(torch.float32)
        x=x.to(device)
        y_target=y_target.to(device)
        
        y_pred, _ = self.model(x)

        x_recon, _ = self.model(y_target,'inverse')

        X_se3_traj = self.inference(x[0,:].unsqueeze(0))

        y_pred = y_pred.cpu().detach().numpy()
        y_target = y_target.cpu().detach().numpy()
        x = x.cpu().detach().numpy()
        x_recon = x_recon.cpu().detach().numpy()

        for i in range(6):
            x[:, i] = x[:, i] * self.dataset.scaling_factors[i]
            x_recon[:, i] = x_recon[:, i] * self.dataset.scaling_factors[i]
            X_se3_traj[:, i] = X_se3_traj[:, i] * self.dataset.scaling_factors[i]

        fig1, axs1 = plt.subplots(6)
        for i in range(6):
            axs1[i].plot(y_target[:,i])
            axs1[i].plot(y_pred[:,i])
        fig2, axs2 = plt.subplots(6)
        for i in range(6):
            axs2[i].plot(x[:,i])
            axs2[i].plot(x_recon[:,i])
        
        x_rot_trajs=self.se3_traj_2_SE3_rot_traj_task(x,goal_H)
        x_recon_rot_trajs=self.se3_traj_2_SE3_rot_traj_task(x_recon[:,:],goal_H)
        X_task_matrix_traj=self.se3_traj_2_SE3_rot_traj_task(X_se3_traj,goal_H)

        fig3 = plt.figure(figsize=(20, 20))
        ax3 = fig3.add_subplot(111, projection='3d')
        ax3.set_xlim([-0.1, 0.4])
        ax3.set_ylim([0.2, 0.7])
        ax3.set_zlim([0.2, 0.7])
        visualize_SE3_traj(x_rot_trajs, ax=ax3, traj_color='blue', traj_linewidth=4.0, frame_scale=0.02, show_frames=True)
        visualize_SE3_traj(x_recon_rot_trajs, ax=ax3, traj_color='red', show_frames=False)
        visualize_SE3_traj(X_task_matrix_traj, ax=ax3, traj_color='green', traj_linewidth=2.0, frame_scale=0.04, show_frames=True)

        plt.show()
        print('Test finished.')

    def inference(self, x):

        from SE3_DS.potential_fields.SE3CanyonPotentialField import SE3CanyonPotentialField

        self.Y_end = np.eye(4)
        self.Canyon=SE3.exp([1,1,1],[1,1,1]).to_matrix()
        self.CPF = SE3CanyonPotentialField(self.Y_end, self.Canyon, alpha=0.1, beta=1, a1=10, a2=1)
        Y , _=self.model(x)
        Y=Y.squeeze(0).detach().cpu().numpy()
        Y=SE3.exp(Y[:3],Y[3:]).to_matrix()
        Y_se3_traj=torch.tensor(self.CPF.inference_canyon_traj(Y, max_len=20000),dtype=torch.float32).to(device)
        X_se3_traj, _=self.model(Y_se3_traj,'inverse')
        X_se3_traj=X_se3_traj.detach().cpu().numpy()

        return X_se3_traj
    
    def jacobian_regularization(self, J):

        s = torch.linalg.svdvals(J)
        condition_loss = (s.max(dim=1)[0] / s.min(dim=1)[0] - 1).mean()
        
        return condition_loss

    def se3_traj_2_SE3_rot_traj(self, se3_traj, goal_H):
        rot_traj = np.zeros((0, 4, 4))
        for t in range(se3_traj.shape[0]):
            H = SE3.exp(se3_traj[t, :3], se3_traj[t, 3:]).to_matrix()
            rot_traj =  np.concatenate((rot_traj, H[None, ...]), 0)
        return rot_traj
    
    def se3_traj_2_SE3_rot_traj_task(self, se3_traj, goal_H):
        rot_traj = np.zeros((0, 4, 4))
        for t in range(se3_traj.shape[0]):
            H = SE3.exp(se3_traj[t, :3], se3_traj[t, 3:]).to_matrix()
            H = np.matmul(goal_H, np.linalg.inv(H))
            rot_traj =  np.concatenate((rot_traj, H[None, ...]), 0)
        return rot_traj


class ConditionalTrainer:
    def __init__(self, model, dataset, batch_size=64, num_epochs=500):
        self.model = model.to(device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def train(self,save_path):
        self.dataloader = DataLoader(self.dataset.dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(),lr=0.005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
        best_train_loss = float('inf')
        for epoch in range(self.num_epochs):
            total_loss1 = 0
            total_loss2 = 0
            total_loss3 = 0
            total_loss = 0
            for _, (x, condition, y_target) in enumerate(self.dataloader):
                if x.dtype != torch.float32:
                    x = x.to(torch.float32)
                    y_target = y_target.to(torch.float32)

                y_pred, J = self.model(x, condition, 'direct')
                loss_forward = F.mse_loss(y_pred, y_target)
                
                x_recon, _ = self.model(y_target, condition, 'inverse')
                loss_backward = F.mse_loss(x_recon, x)
                
                loss_jacobian = 0.0005*self.jacobian_regularization(J)
                loss1=loss_forward
                loss2=loss_backward
                loss3=loss_jacobian
                loss = loss_forward + loss_backward + loss_jacobian
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            total_loss3 += loss3.item()
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {total_loss}, Loss1: {total_loss1}, Loss2: {total_loss2}, Loss3: {total_loss3}, LR: {scheduler.get_last_lr()[0]}')
            
            if total_loss < best_train_loss:
                best_train_loss = total_loss
                torch.save(self.model.state_dict(), save_path)
                print(f'Saved best model to {save_path}')

    def test(self,model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.dataloader = DataLoader(self.dataset.dataset, batch_size=self.dataset.dataset.traj_len, shuffle=False)
        x, condition, y_target = next(iter(self.dataloader))
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
            y_target = y_target.to(torch.float32)
        x=x.to(device)
        y_target=y_target.to(device)

        x_recon, _ = self.model(y_target, condition, 'inverse')

        x = x.cpu().detach().numpy()
        x_recon = x_recon.cpu().detach().numpy()

        for i in range(4):
            for j in range(6):
                x[1500*i:1500*(i+1), j] = x[1500*i:1500*(i+1), j] * self.dataset.scaling_factors[j]
                x_recon[1500*i:1500*(i+1), j] = x_recon[1500*i:1500*(i+1), j] * self.dataset.scaling_factors[j]

        x=self.se3_traj_2_SE3_rot_traj_task(x,self.dataset.goal_H)
        x_recon=self.se3_traj_2_SE3_rot_traj_task(x_recon,self.dataset.goal_H)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(4):
            visualize_SE3_traj(x[1500*i:1500*(i+1),:], ax=ax, traj_color='blue',traj_linewidth=4.0, frame_scale=0.02, show_frames=True)
            visualize_SE3_traj(x_recon[1500*i:1500*(i+1),:], ax=ax, traj_color='red',traj_linewidth=2.0, frame_scale=0.04, show_frames=True)

        target=torch.zeros((1000,6), dtype=torch.float32).to(device)
        weights = torch.linspace(1, 0, 1000).unsqueeze(1)
        target[:,:]= weights

        fig2 = plt.figure()
        ax = fig2.add_subplot(111, projection='3d')
        for i in range(4):
            visualize_SE3_traj(x[1500*i:1500*(i+1),:], ax=ax, traj_color='blue',traj_linewidth=4.0, frame_scale=0.02, frame_interval=100, show_frames=True)
        
        cvalues=np.linspace(0.25,1,20)
        for i in range(20):
            cvalue=cvalues[i]
            condition=cvalue*torch.ones((1000,1), dtype=torch.float32).to(device)
            x_recon, _ = self.model(target, condition, 'inverse')
            x_recon = x_recon.cpu().detach().numpy()
            for j in range(6):
                x_recon[:, j] = x_recon[:, j] * self.dataset.scaling_factors[j]
            x_recon=self.se3_traj_2_SE3_rot_traj_task(x_recon,self.dataset.goal_H)
            visualize_SE3_traj(x_recon, ax=ax, traj_color='red',traj_linewidth=2.0, frame_scale=0.02, frame_interval=100, show_frames=True)

        plt.show()
        print('Test finished.')
    
    def jacobian_regularization(self, J):
        s = torch.linalg.svdvals(J)
        condition_loss = (s.max(dim=1)[0] / s.min(dim=1)[0] - 1).mean()
        
        return condition_loss
    def se3_traj_2_SE3_rot_traj_task(self, se3_traj, goal_H):
        rot_traj = np.zeros((0, 4, 4))
        for t in range(se3_traj.shape[0]):
            H = SE3.exp(se3_traj[t, :3], se3_traj[t, 3:]).to_matrix()
            H = np.matmul(goal_H, np.linalg.inv(H))
            rot_traj =  np.concatenate((rot_traj, H[None, ...]), 0)
        return rot_traj