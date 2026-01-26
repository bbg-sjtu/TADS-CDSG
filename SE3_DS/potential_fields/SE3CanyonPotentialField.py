import numpy as np
from sophus_pybind import SE3
from SE3_DS.utils.utils import *


class SE3CanyonPotentialField:
    """
    SE(3) TADS
    """
    def __init__(self, Y_end, Canyon, alpha=0.1, beta=1, a1=10, a2=1):

        self.alpha = alpha
        self.beta = beta
        self.a1 = a1
        self.a2 = a2
        self.W = np.diag([alpha]*3 + [beta]*3)
        self.W_inv = np.diag([1/alpha]*3 + [1/beta]*3)

        self.Y_end=Y_end
        self.Canyon=Canyon
        self.eta_i=self.get_eta()

    def exp_se3(self, xi):
        return SE3.exp(xi[:3], xi[3:]).to_matrix()
    def log_se3(self, T):
        return T.log().flatten()
    def hat_se3(self, xi):
        v, omega = xi[:3], xi[3:]
        Omega = self.hat_so3(omega)
        return np.block([
            [Omega, v.reshape(3,1)],
            [np.zeros((1,4))]
        ])
    def vee_se3(self, xi_hat):
        return np.array([xi_hat[0,3], xi_hat[1,3], xi_hat[2,3],
                         xi_hat[2,1], xi_hat[0,2], xi_hat[1,0]])
    def hat_so3(self, omega):
        return np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0]
        ])
    
    def inner_product(self, xi1, xi2, clip_value=1e3):
        xi1 = np.clip(xi1, -clip_value, clip_value)
        xi2 = np.clip(xi2, -clip_value, clip_value)
        result = xi1 @ self.W @ xi2
        return np.clip(result, -1e10, 1e10)
    
    def norm_se3(self, xi):
        return np.sqrt(self.inner_product(xi, xi))
    
    def dexp_matrix(self, xi):
        if np.linalg.norm(xi[3:]) < 1e-8:
            return np.eye(6)
        
        v = xi[:3]
        w = xi[3:]
        theta = np.linalg.norm(w)
        
        w_hat = self.hat_so3(w)
        w_hat_sq = w_hat @ w_hat
        
        A = np.eye(3) + (1 - np.cos(theta)) / (theta**2) * w_hat + (theta - np.sin(theta)) / (theta**3) * w_hat_sq
        
        v_hat = self.hat_so3(v)
        
        B = (1/2*np.eye(3) + (theta - np.sin(theta)) / (theta**3) * w_hat + ((theta**2 + 2 * np.cos(theta) - 2) / (2 * theta**4)) * w_hat_sq) @ v_hat
        
        dexp_mat = np.zeros((6, 6))
        dexp_mat[:3, :3] = A
        dexp_mat[:3, 3:] = B
        dexp_mat[3:, 3:] = A
        
        return dexp_mat
    
    def dlog_matrix(self, X):
        log_X = self.log_se3(X)
        dexp_mat = self.dexp_matrix(log_X)
        try:
            dlog_mat = np.linalg.inv(dexp_mat)
        except np.linalg.LinAlgError:
            dlog_mat = np.linalg.pinv(dexp_mat)
        
        return dlog_mat
    
    def dlog_adjoint(self, X, xi):
        dlog_mat = self.dlog_matrix(X)
        dlog_adj = self.W_inv @ dlog_mat.T @ self.W
        dlog_adj = np.clip(dlog_adj, -1e4, 1e4)
        xi = np.clip(xi, -1e4, 1e4)
        result = dlog_adj @ xi
        return np.clip(result, -1e10, 1e10)
    
    def se3_adjoint_action(self, M, tau):

        R = M[0:3, 0:3]
        t = M[0:3, 3]

        t_skew = np.array([
            [0, -t[2], t[1]],
            [t[2], 0, -t[0]],
            [-t[1], t[0], 0]
        ])

        Ad_M = np.zeros((6, 6))
        Ad_M[0:3, 0:3] = R
        Ad_M[0:3, 3:6] = t_skew @ R
        Ad_M[3:6, 3:6] = R

        transformed_tau = Ad_M @ tau
        
        return transformed_tau

    def natural_gradient_descent(self, Y):

        X = np.linalg.inv(Y) @ self.Y_end # Y^{-1}Y*
        X_se3 = SE3.from_matrix(X)
        xi = self.log_se3(X_se3)  # ξ = log(Y^{-1}Y*)
        mu = -self.dlog_adjoint(X_se3, xi)  # μ = -(dlog_X)^*(ξ)
        return -self.se3_adjoint_action(Y, mu), -mu
    
    def canyon_potential_gradient(self, Y):
        """
        V(Y) = 1/2 * a1 * ||ξ||^2 * θ^2 + 1/2 * a2 * ||ξ||^2
        grad V(Y) = Y * [(-a1θ^2 - a2) * (dlog_X)^*(ξ) - a1 * ||ξ||^2 * θ * A(ξ, η_i)]
        """
        X = np.linalg.inv(Y) @ self.Y_end
        X_se3 = SE3.from_matrix(X)
        
        xi = self.log_se3(X_se3)  # ξ = log(Y^{-1}Y*)
        
        xi_norm = self.norm_se3(xi)
        eta_norm = self.norm_se3(self.eta_i)
        
        if xi_norm < 1e-8 or eta_norm < 1e-8:
            return self.natural_gradient_descent(Y)
        
        cos_theta = self.inner_product(xi, self.eta_i) / (xi_norm * eta_norm)
        cos_theta = np.clip(cos_theta, -1, 1)
        theta = np.arccos(cos_theta)
        
        dlog_star_xi = self.dlog_adjoint(X_se3, xi)
        dlog_star_eta = self.dlog_adjoint(X_se3, self.eta_i)
        
        # A(ξ, η_i)
        if theta < 1e-8:
            A_term = np.zeros(6)
        else:
            numerator = dlog_star_eta * xi_norm - (self.inner_product(xi, self.eta_i) / xi_norm) * dlog_star_xi
            denominator = xi_norm**2 * eta_norm * np.sqrt(1 - cos_theta**2)
            eps=1e-10
            denominator = np.where(np.abs(denominator) < eps, 
                                np.sign(denominator) * eps, 
                                denominator)
            A_term = numerator / denominator
        
        # μ
        mu = (-self.a1 * theta**2 - self.a2) * dlog_star_xi + self.a1 * xi_norm**2 * theta * A_term
        # grad = Y @ mu
        return -self.se3_adjoint_action(Y, mu), -mu
    
    def get_eta(self):
        C_i_inv_Y = np.linalg.inv(self.Canyon) @ self.Y_end
        C_i_inv_Y_se3 = SE3.from_matrix(C_i_inv_Y)
        return self.log_se3(C_i_inv_Y_se3)  # η_i = log(C_i^{-1}Y*)

    def inference_canyon_traj(self, Y_start, max_len=1500):
        Y_traj=[]
        Y_traj.append(SE3.from_matrix(Y_start).log()[0])
        Y=Y_start
        for i in range(max_len-1):
            v_Y, _ = self.canyon_potential_gradient(Y)
            new_Y=self.apply_velocity_command(Y, v_Y)
            Y_traj.append(SE3.from_matrix(new_Y).log()[0])
            Y=new_Y
            if np.linalg.norm(Y-np.eye(4))<1e-3:
                break
        print(len(Y_traj))
        return np.array(Y_traj)
    
    def apply_velocity_command(self, Y, velocity_command):
        dt=0.002

        current_position=Y[:3,3]
        current_rotation=Y[:3,:3]

        v = velocity_command[0:3]
        w = velocity_command[3:6]

        new_position = current_position + v * dt
        
        w_norm = np.linalg.norm(w)
        if w_norm > 1e-10:
            axis = w / w_norm
            angle = w_norm * dt
            
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            one_minus_cos = 1 - cos_a
            
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            
            R_delta = np.eye(3) + sin_a * K + one_minus_cos * (K @ K)
            
            new_rotation = R_delta @ current_rotation
        else:
            new_rotation = current_rotation
        
        new_Y = np.eye(4)
        new_Y[:3,:3] = new_rotation
        new_Y[:3,3] = new_position
        
        return new_Y
    