import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from queue import Queue
from sophus_pybind import SE3
from potential_fields.SE3CanyonPotentialField import SE3CanyonPotentialField
from dataset.dataset import SE3_DATASET
from nn_models.flows import BijectionNet
import torch
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

class RealTimeCoordinateVisualizer:
    def __init__(self, Y_end, Y_start, C, X_end, X_start, teach_data, CPF:SE3CanyonPotentialField):

        self.Y_end = Y_end
        self.Y_start = Y_start
        self.C = C
        self.X_end = X_end
        self.X_start = X_start
        self.teach_data=teach_data
        self.teach_data=teach_data.norm_rot_trajs[0]
        self.scaling_factors=teach_data.scaling_factors

        self.coordinate_systems = []
        self.trajectories = []
        self.pose_traces = []

        self.current_time = 0
        self.dt = 0.01
        self.is_running = True
        self.CPF = CPF

        self.pose_trace_interval = 0.5
        self.last_pose_trace_time = 0
        self.pose_trace_alpha = 0.4
        self.pose_trace_scale = 0.3
        self.velocity_queue = Queue()

        self.fig = plt.figure(figsize=(12, 10))
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax2 = self.fig.add_subplot(122, projection='3d')

        self.ax1.set_xlim([0, 1.5])
        self.ax1.set_ylim([0, 1.5])
        self.ax1.set_zlim([0, 1.5])
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        self.ax1.set_title('Latent Space')

        self.ax2.set_xlim([-0.5, 0.1])
        self.ax2.set_ylim([-0.1, 0.5])
        self.ax2.set_zlim([0, 0.6])
        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('Y')
        self.ax2.set_zlabel('Z')
        self.ax2.set_title('Task Space')

        self.Y_state_record=[]
        
        self.visualize_geodesic_latentspace(self.Y_end, self.C)
        self.visualize_teach_data()

        self.add_coordinate_from_pose(self.ax1, Y_end, "Target", "red",axis_length=0.5)
        self.add_coordinate_from_pose(self.ax1, Y_start, "Y", "green", axis_length=0.5, add_initial_trace=True)
        self.add_coordinate_from_pose(self.ax1, C, "Canyon", "blue", axis_length=0.5)
        self.add_coordinate_from_pose(self.ax2, X_end, "Target", "red", axis_length=0.1)
        self.add_coordinate_from_pose(self.ax2, X_start, "X", "green", axis_length=0.1, add_initial_trace=True)

    def visualize_geodesic_latentspace(self, Y_end, C):
        xi = CPF.log_se3(SE3.from_matrix(np.linalg.inv(C) @ Y_end))
        path = []
        for t in np.linspace(0, 1, 100):
            T_t = C @ CPF.exp_se3(t * xi)
            path.append(T_t)

        positions = [T[:3, 3] for T in path]
        positions = np.array(positions)
        
        self.ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', alpha=0.7, label='轨迹')
        
        step = max(1, len(path) // 10)
        for i in range(0, len(path), step):
            alpha = 0.3 + 0.7 * (i / len(path))
            self.plot_frame(self.ax1, path[i], scale=0.1, alpha=alpha)

    def visualize_teach_data(self):
        path=self.teach_data

        positions = [T[:3, 3] for T in path]
        positions = np.array(positions)
        
        self.ax2.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', alpha=0.7, label='轨迹')
        
        step = max(1, len(path) // 10)
        for i in range(0, len(path), step):
            alpha = 0.3 + 0.7 * (i / len(path))
            self.plot_frame(self.ax2, path[i], scale=0.05, alpha=alpha)

    def plot_frame(self, ax, T, scale=1.0, label="", alpha=1.0):

        origin = T[:3, 3]
        x_axis = T[:3, :3] @ np.array([scale, 0, 0])
        y_axis = T[:3, :3] @ np.array([0, scale, 0])
        z_axis = T[:3, :3] @ np.array([0, 0, scale])

        ax.quiver(*origin, *x_axis, color='r', alpha=alpha, arrow_length_ratio=0.1)
        ax.quiver(*origin, *y_axis, color='g', alpha=alpha, arrow_length_ratio=0.1)
        ax.quiver(*origin, *z_axis, color='b', alpha=alpha, arrow_length_ratio=0.1)

        if label:
            ax.text(*origin, label, fontsize=12)
    
    def init_world_coordinate(self):

        origin = np.array([0, 0, 0])
        rotation = np.eye(3)
        self.add_coordinate_system(self.ax1, origin, rotation, 'Target', 'black', axis_length=0.5)
        self.add_coordinate_system(self.ax2, origin, rotation, 'Target', 'black', axis_length=0.1)

    def add_coordinate_system(self, ax, position, rotation, name, label_color, axis_length):

        x_axis = rotation @ np.array([1, 0, 0])
        y_axis = rotation @ np.array([0, 1, 0])
        z_axis = rotation @ np.array([0, 0, 1])
        
        x_line = ax.quiver(*position, *x_axis, color='red', length=axis_length, 
                               normalize=True, linewidth=2, arrow_length_ratio=0.1)
        y_line = ax.quiver(*position, *y_axis, color='green', length=axis_length, 
                               normalize=True, linewidth=2, arrow_length_ratio=0.1)
        z_line = ax.quiver(*position, *z_axis, color='blue', length=axis_length, 
                               normalize=True, linewidth=2, arrow_length_ratio=0.1)

        label_pos = position + 0.6 * axis_length * (x_axis + y_axis + z_axis) / 3
        ax.text(*label_pos, name, color=label_color, fontsize=20, fontweight='bold')

        coord_sys = {
            'position': position.copy(),
            'rotation': rotation.copy(),
            'name': name,
            'label_color': label_color,
            'quivers': [x_line, y_line, z_line],
            'trajectory': [position.copy()],
            'velocity': np.zeros(6),
            'pose_traces': []
        }
        self.coordinate_systems.append(coord_sys)
        
        trajectory_line, = ax.plot([], [], [], color=label_color, linewidth=2, alpha=0.7)
        self.trajectories.append(trajectory_line)
        
        return len(self.coordinate_systems) - 1
    
    def add_pose_trace(self, ax, coord_index, position, rotation, axis_length):

        coord_sys = self.coordinate_systems[coord_index]

        x_axis = rotation @ np.array([1, 0, 0])
        y_axis = rotation @ np.array([0, 1, 0])
        z_axis = rotation @ np.array([0, 0, 1])
        
        axis_length = axis_length * self.pose_trace_scale

        x_trace = ax.quiver(*position, *x_axis, color='red', length=axis_length, 
                                normalize=True, linewidth=1, arrow_length_ratio=0.1, alpha=self.pose_trace_alpha)
        y_trace = ax.quiver(*position, *y_axis, color='green', length=axis_length, 
                                normalize=True, linewidth=1, arrow_length_ratio=0.1, alpha=self.pose_trace_alpha)
        z_trace = ax.quiver(*position, *z_axis, color='blue', length=axis_length, 
                                normalize=True, linewidth=1, arrow_length_ratio=0.1, alpha=self.pose_trace_alpha)
        
        trace = {
            'quivers': [x_trace, y_trace, z_trace],
            'position': position.copy(),
            'rotation': rotation.copy()
        }
        
        coord_sys['pose_traces'].append(trace)
        
        if len(coord_sys['pose_traces']) > 50:

            old_trace = coord_sys['pose_traces'].pop(0)
            for quiver in old_trace['quivers']:
                quiver.remove()
    
    def pose_matrix_to_position_rotation(self, pose_matrix):

        if pose_matrix.shape == (4, 4):
            position = pose_matrix[:3, 3]
            rotation = pose_matrix[:3, :3]
        elif pose_matrix.shape == (3, 4):
            position = pose_matrix[:, 3]
            rotation = pose_matrix[:3, :3]
        else:
            raise ValueError("Pose matrix should be 3x4 or 4x4")
        
        return position, rotation
    
    def add_coordinate_from_pose(self, ax, pose_matrix, name, label_color, axis_length, add_initial_trace=False):

        position, rotation = self.pose_matrix_to_position_rotation(pose_matrix)
        index = self.add_coordinate_system(ax, position, rotation, name, label_color, axis_length)
        
        if add_initial_trace:
            self.add_pose_trace(ax, index, position, rotation, axis_length)
            
        return index
    
    def update_coordinate_system(self, ax, index, new_position, new_rotation):

        coord_sys = self.coordinate_systems[index]

        old_position = coord_sys['position']
        coord_sys['position'] = new_position.copy()
        coord_sys['rotation'] = new_rotation.copy()

        if np.linalg.norm(new_position - old_position) > 0.001:
            coord_sys['trajectory'].append(new_position.copy())
        
        if len(coord_sys['trajectory']) > 1000:
            coord_sys['trajectory'] = coord_sys['trajectory'][-500:]
        
        x_axis = new_rotation @ np.array([1, 0, 0])
        y_axis = new_rotation @ np.array([0, 1, 0])
        z_axis = new_rotation @ np.array([0, 0, 1])
        
        axis_length = 0.1
        
        for quiver in coord_sys['quivers']:
            quiver.remove()
        
        x_line = ax.quiver(*new_position, *x_axis, color='red', 
                               length=axis_length, normalize=True, linewidth=2, 
                               arrow_length_ratio=0.1)
        y_line = ax.quiver(*new_position, *y_axis, color='green', 
                               length=axis_length, normalize=True, linewidth=2, 
                               arrow_length_ratio=0.1)
        z_line = ax.quiver(*new_position, *z_axis, color='blue', 
                               length=axis_length, normalize=True, linewidth=2, 
                               arrow_length_ratio=0.1)
        
        coord_sys['quivers'] = [x_line, y_line, z_line]
        
        if len(coord_sys['trajectory']) > 1:
            trajectory = np.array(coord_sys['trajectory'])
            self.trajectories[index].set_data(trajectory[:, 0], trajectory[:, 1])
            self.trajectories[index].set_3d_properties(trajectory[:, 2])
    
    def apply_velocity_command(self, ax, coord_index, axis_length, velocity_command):

        coord_sys = self.coordinate_systems[coord_index]

        coord_sys['velocity'] = np.array(velocity_command)

        v = np.array(velocity_command[0:3])
        w = np.array(velocity_command[3:6])

        new_position = coord_sys['position'] + v * self.dt

        w_norm = np.linalg.norm(w)
        if w_norm > 1e-10:

            axis = w / w_norm
            angle = w_norm * self.dt
            
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            one_minus_cos = 1 - cos_a
            
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            
            R_delta = np.eye(3) + sin_a * K + one_minus_cos * (K @ K)

            new_rotation = R_delta @ coord_sys['rotation']
        else:
            new_rotation = coord_sys['rotation']
        
        self.update_coordinate_system(ax, coord_index, new_position, new_rotation)
        
        if coord_index == 2 and self.current_time - self.last_pose_trace_time >= self.pose_trace_interval:
            self.add_pose_trace(ax, coord_index, new_position, new_rotation, axis_length)
            self.last_pose_trace_time = self.current_time
        
        return new_position, new_rotation
    
    def set_velocity_command(self, coord_index, velocity_command):
        self.velocity_queue.put((coord_index, velocity_command))
    
    def get_coordinate_state(self, coord_index):
        if 0 <= coord_index < len(self.coordinate_systems):
            coord_sys = self.coordinate_systems[coord_index]
            return {
                'position': coord_sys['position'].copy(),
                'rotation': coord_sys['rotation'].copy(),
                'velocity': coord_sys['velocity'].copy(),
                'time': self.current_time
            }
        return None
    
    def animate(self, frame):
        """动画更新函数"""
        if not self.is_running:
            return []

        while not self.velocity_queue.empty():
            try:
                coord_index, velocity_command = self.velocity_queue.get_nowait()
                if 0 <= coord_index < len(self.coordinate_systems):
                    self.apply_velocity_command(coord_index, velocity_command)
            except:
                pass
        
        try:

            Y_state = self.get_coordinate_state(1)
            X_state = self.get_coordinate_state(4)
            
            if Y_state is not None:
                v_Y = self.CPF_control_callback(Y_state, X_state)
                if v_Y is not None and len(v_Y) == 6:
                    self.apply_velocity_command(self.ax1, 1, 0.5, v_Y)

        except Exception as e:
            print(f"控制回调函数错误: {e}")
        
        self.current_time += self.dt

        self.fig.canvas.draw_idle()
        return []
    
    def start_animation(self):
        print("Start animation...")

        self.anim = animation.FuncAnimation(
            self.fig, self.animate, frames=None,
            interval=self.dt * 1000, blit=False, repeat=True, cache_frame_data=False
        )
        
        try:
            plt.show()
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        self.is_running = False
        if hasattr(self, 'anim'):
            self.anim.event_source.stop()
        plt.close()


    def CPF_control_callback(self, Y_state, X_state):

        Y_ = np.eye(4)
        Y_[:3, :3] = Y_state['rotation']
        Y_[:3, 3] = Y_state['position']

        X_ = np.eye(4)
        X_[:3, :3] = X_state['rotation']
        X_[:3, 3] = X_state['position']

        Y_se3=SE3.from_matrix(Y_).log()
        Y_se3=torch.tensor(Y_se3,dtype=torch.float32)
        X_se3, J = diffieo(Y_se3,'inverse')
        X_se3=X_se3.squeeze().detach().numpy()
        X_se3=X_se3*self.scaling_factors
        X=SE3.exp(X_se3[:3],X_se3[3:]).to_matrix()

        self.update_coordinate_system(self.ax2, 4, X[:3,3], X[:3,:3])

        v_Y_space, v_Y_se3 = self.CPF.canyon_potential_gradient(Y_)

        return v_Y_space
    
    def phi(self, Y):
        Y_se3=SE3.from_matrix(Y).log()
        Y_se3=torch.tensor(Y_se3,dtype=torch.float32)
        X_se3, J = diffieo(Y_se3,'inverse')
        X_se3=X_se3.squeeze().detach().numpy()
        X=SE3.exp(X_se3[:3],X_se3[3:]).to_matrix()
        return X
    
    def compute_jacobian_autodiff(self, y):

        y_tensor = torch.tensor(y, dtype=torch.float32, requires_grad=True)
        
        x_tensor,_ = diffieo(y_tensor,'inverse')

        J = torch.zeros(6, 6)

        for i in range(6):

            grad_output = torch.zeros(6)
            grad_output[i] = 1.0

            if y_tensor.grad is not None:
                y_tensor.grad.zero_()

            x_tensor=x_tensor.squeeze(0)
            x_tensor.backward(gradient=grad_output, retain_graph=True)

            J[i, :] = y_tensor.grad.clone()
        
        return J.detach().numpy()

    def compute_jacobian_finite_difference(self, y, epsilon=1e-6):

            y_tensor = torch.tensor(y, dtype=torch.float32)
            
            x_base,_ = diffieo(y_tensor, 'inverse')
            x_base=x_base.detach().numpy()

            J = np.zeros((6, 6))

            for j in range(6):

                y_perturbed = y.copy()
                y_perturbed[:,j] += epsilon
                
                y_pert_tensor = torch.tensor(y_perturbed, dtype=torch.float32)
                
                x_perturbed,_ = diffieo(y_pert_tensor, 'inverse')
                x_perturbed=x_perturbed.detach().numpy()

                J[:, j] = (x_perturbed - x_base) / epsilon
            
            return J

device='cpu'
Y_end = np.eye(4)
Canyon=SE3.exp([1,1,1],[1,1,1]).to_matrix()
CPF = SE3CanyonPotentialField(Y_end, Canyon, alpha=0.1, beta=1, a1=10, a2=1)

# Data preparation
_THIS_DIR = Path(__file__).resolve().parent
data_name='000'
data_path = _THIS_DIR / f"dataset/PegInHole_data/teach_data_{data_name}.txt"
model_path = _THIS_DIR / f"nn_models/models/PegInHole_models/best_model_{data_name}.pth"
data=SE3_DATASET(data_path)
diffieo=BijectionNet(num_dims=6, num_blocks=5, num_hidden=64, device=device)
diffieo.load_state_dict(torch.load(model_path))

# Latent space
# 0. Target
Y_end = np.eye(4)
# 1 Controllable Y
Y_start=SE3.exp([1,1,1],[1,1,1]).to_matrix()
# 2. Canyon C
C=SE3.exp([1,1,1],[1,1,1]).to_matrix()

# Task space
# 3.Target
X_end = np.eye(4)
# 4.Controllable X
Y_se3=torch.tensor(SE3.from_matrix(Y_start).log()[0],dtype=torch.float32).unsqueeze(0)
X_start, _ = diffieo(Y_se3, 'inverse')
X_start=X_start.squeeze().detach().numpy()
X_start=X_start*data.scaling_factors
X_start=SE3.exp(X_start[:3],X_start[3:]).to_matrix()

visualizer = RealTimeCoordinateVisualizer(Y_end, Y_start, C, X_end, X_start, data, CPF)

visualizer.start_animation()