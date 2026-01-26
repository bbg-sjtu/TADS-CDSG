import numpy as np
from scipy.spatial.transform import Rotation as R

def xyz_rotvec_2_matrix(pose):
    translation = pose[:3]
    rot_vec = pose[3:]
    angle = np.linalg.norm(rot_vec)
    if angle < 1e-8:
        rotation_matrix = np.eye(3)
    else:
        rotation = R.from_rotvec(rot_vec)
        rotation_matrix = rotation.as_matrix()
    H = np.eye(4)
    H[:3, :3] = rotation_matrix
    H[:3, 3] = translation
    return H

def matrix_2_xyz_rotvec(matrix):
    translation = matrix[:3, 3]
    rotation_matrix = matrix[:3, :3]
    rotation = R.from_matrix(rotation_matrix)
    rot_vec = rotation.as_rotvec()
    pose = np.zeros(6)
    pose[:3] = translation
    pose[3:] = rot_vec
    return pose

def matrix_2_xyz_rotvec_batch(matrix_traj):
    pose_traj=np.zeros((matrix_traj.shape[0],6))
    for i in range(matrix_traj.shape[0]):
        translation = matrix_traj[i, :3, 3]
        rotation_matrix = matrix_traj[i, :3, :3]
        rotation = R.from_matrix(rotation_matrix)
        rot_vec = rotation.as_rotvec()
        pose = np.zeros(6)
        pose[:3] = translation
        pose[3:] = rot_vec
        pose_traj[i,:]=pose
    return pose_traj


def visualize_SE3_traj(x_rot_trajs, ax, frame_interval=50, frame_scale=0.01, 
                      traj_color='blue', traj_linewidth=2, show_frames=True):
    positions = x_rot_trajs[:, :3, 3]
    rotations = x_rot_trajs[:, :3, :3]
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            color=traj_color, linewidth=traj_linewidth, label='traj')
    if show_frames and frame_interval > 0:
        n_frames = len(x_rot_trajs)
        indices = np.arange(0, n_frames, frame_interval)
        if len(indices) < 2 and n_frames > 1:
            indices = [0, n_frames-1]
        for idx in indices:
            pos = positions[idx]
            rot = rotations[idx]
            origin = pos
            x_axis = rot @ np.array([frame_scale, 0, 0])
            y_axis = rot @ np.array([0, frame_scale, 0])
            z_axis = rot @ np.array([0, 0, frame_scale])
            ax.quiver(*origin, *x_axis, color='r', alpha=0.8, arrow_length_ratio=0.1, lw=traj_linewidth)
            ax.quiver(*origin, *y_axis, color='g', alpha=0.8, arrow_length_ratio=0.1, lw=traj_linewidth)
            ax.quiver(*origin, *z_axis, color='b', alpha=0.8, arrow_length_ratio=0.1, lw=traj_linewidth)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True, alpha=0.3)





