#!/usr/bin/env python
import rospy
import tf
import numpy as np
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Vector3

# --- CONSTANTS ---
STD_GYRO = 0.002
STD_ACCEL = 0.04
STD_CONTACT = 0.2
STD_PRIOR_FWD_KIN = 0.03

# Debugging Thresholds
VELOCITY_ERROR_THRESHOLD = 0.5 

class InEKF:
    def __init__(self):
        # State X (7x7): [R, v, p, dR, dL]
        self.X = np.eye(7)
        self.P = np.eye(21) * 0.1
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.N_fixed = np.eye(3) * (STD_CONTACT**2 + STD_PRIOR_FWD_KIN**2)
        
        self.g_vector = np.array([0, 0, -9.81]) 
        self.is_initialized = False

    def skew(self, v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    def exp_so3(self, phi):
        theta = np.linalg.norm(phi)
        Phi_skew = self.skew(phi)
        if theta < 1e-6:
            return np.eye(3) + Phi_skew + 0.5 * np.dot(Phi_skew, Phi_skew)
        else:
            return np.eye(3) + (np.sin(theta)/theta)*Phi_skew + ((1-np.cos(theta))/theta**2)*np.dot(Phi_skew, Phi_skew)

    def exp_se_k(self, v):
        K = (len(v) - 3) // 3
        phi = v[0:3]
        theta = np.linalg.norm(phi)
        Phi_skew = self.skew(phi)
        
        if theta < 1e-6:
            R = np.eye(3) + Phi_skew + 0.5 * np.dot(Phi_skew, Phi_skew)
            V_left = np.eye(3) + 0.5 * Phi_skew + (1.0/6.0) * np.dot(Phi_skew, Phi_skew)
        else:
            A2 = np.dot(Phi_skew, Phi_skew)
            R = np.eye(3) + (np.sin(theta)/theta)*Phi_skew + ((1-np.cos(theta))/theta**2)*A2
            V_left = np.eye(3) + ((1-np.cos(theta))/theta**2)*Phi_skew + ((theta-np.sin(theta))/theta**3)*A2
        
        dim = 3 + K
        X = np.eye(dim)
        X[0:3, 0:3] = R
        for i in range(K):
            X[0:3, 3+i] = np.dot(V_left, v[3+3*i : 3+3*(i+1)])
        return X

    def predict(self, w, a, dt, left_contact, right_contact, y_left, y_right):
        """
        w, a: IMU inputs (Already rotated to Base Frame)
        y_left/right: Kinematic positions of feet relative to Base
        """
        # 1. Bias Correct
        w_k = w - self.bg
        a_k = a - self.ba
        
        R = self.X[0:3, 0:3]
        v = self.X[0:3, 3]
        p = self.X[0:3, 4]
        dR = self.X[0:3, 5]
        dL = self.X[0:3, 6]

        # 2. Propagate Base State
        R_pred = np.dot(R, self.exp_so3(w_k * dt))
        acc_world = np.dot(R, a_k) + self.g_vector
        v_pred = v + acc_world * dt
        p_pred = p + v * dt + 0.5 * acc_world * dt**2

        # 3. Propagate Feet (Track kinematics if in air)
        # d_world = p_world + R_world * y_body
        
        # Right Foot
        dR_kin = p_pred + np.dot(R_pred, y_right)
        dR_pred = dR if right_contact else dR_kin
        
        # Left Foot
        dL_kin = p_pred + np.dot(R_pred, y_left)
        dL_pred = dL if left_contact else dL_kin

        self.X[0:3, 0:3] = R_pred
        self.X[0:3, 3] = v_pred
        self.X[0:3, 4] = p_pred
        self.X[0:3, 5] = dR_pred
        self.X[0:3, 6] = dL_pred

        # 4. Propagate Covariance
        A = np.zeros((21, 21))
        A[3:6, 0:3] = self.skew(self.g_vector)
        A[6:9, 3:6] = np.eye(3)
        
        A[0:3, 15:18] = -R
        A[3:6, 15:18] = -np.dot(self.skew(v), R)
        A[3:6, 18:21] = -R
        A[6:9, 15:18] = -np.dot(self.skew(p), R)
        A[9:12, 15:18] = -np.dot(self.skew(dR), R)
        A[12:15, 15:18] = -np.dot(self.skew(dL), R)

        Phi = np.eye(21) + A * dt
        Q = np.eye(21) * 0.001
        Q[0:3, 0:3] *= STD_GYRO**2
        Q[3:6, 3:6] *= STD_ACCEL**2
        
        self.P = np.dot(np.dot(Phi, self.P), Phi.T) + Q * dt

    def update(self, y_meas, foot_index):
        R = self.X[0:3, 0:3]
        p = self.X[0:3, 4]
        H = np.zeros((3, 21))
        
        if foot_index == 0: # Right
            d = self.X[0:3, 5]
            H[0:3, 6:9] = -np.eye(3)
            H[0:3, 9:12] = np.eye(3)
        else: # Left
            d = self.X[0:3, 6]
            H[0:3, 6:9] = -np.eye(3)
            H[0:3, 12:15] = np.eye(3)

        # --- REVERTED SIGN (STABILITY FIX) ---
        # Z = p + R*y - d
        # This aligns with the Right-Invariant Error update for position
        z = p + np.dot(R, y_meas) - d
        
        S = np.dot(np.dot(H, self.P), H.T) + self.N_fixed
        try:
            K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        except np.linalg.LinAlgError:
            return

        delta = np.dot(K, z)
        dX = self.exp_se_k(delta[0:15])
        self.X = np.dot(dX, self.X)
        self.bg += delta[15:18]
        self.ba += delta[18:21]
        
        I = np.eye(21)
        temp = (I - np.dot(K, H))
        self.P = np.dot(np.dot(temp, self.P), temp.T) + np.dot(np.dot(K, self.N_fixed), K.T)

class InEKF_ROS:
    def __init__(self):
        rospy.init_node('inekf_estimator', anonymous=True)
        self.filter = InEKF()
        
        self.base_frame = 'base_link'
        self.imu_frame = 'imu_link'
        self.left_foot_frame = 'left_sole_link'
        self.right_foot_frame = 'right_sole_link'
        self.odom_frame = 'odom'
        
        self.left_contact = False
        self.right_contact = False
        self.contact_threshold = 200.0
        self.last_imu_time = None
        
        self.init_buffer = []
        self.init_buffer_size = 50 
        self.gt_vel = None
        self.gt_time = 0.0
        
        self.listener = tf.TransformListener()
        self.pub_odom = rospy.Publisher('/base_vel', Odometry, queue_size=10)
        
        # --- IMU Extrinsics (Base <- IMU) ---
        self.R_imu_base = np.eye(3) 
        self.extrinsics_loaded = False
        
        rospy.Subscriber('/base_imu', Imu, self.imu_callback)
        rospy.Subscriber('/joint_states', JointState, self.joint_callback)
        rospy.Subscriber('/simulator/floating_base_state', Odometry, self.gt_callback)
        
        rospy.loginfo("InEKF Initializing... Waiting for TF.")

    def load_extrinsics(self):
        try:
            self.listener.waitForTransform(self.base_frame, self.imu_frame, rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform(self.base_frame, self.imu_frame, rospy.Time(0))
            self.R_imu_base = tf.transformations.quaternion_matrix(rot)[:3, :3]
            self.extrinsics_loaded = True
            rospy.loginfo("Extrinsics Loaded! R_imu_base:\n%s", str(self.R_imu_base))
            return True
        except (tf.Exception):
            rospy.logwarn_throttle(2, "Waiting for TF: %s -> %s" % (self.base_frame, self.imu_frame))
            return False

    def gt_callback(self, msg):
        self.gt_vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
        self.gt_time = msg.header.stamp.to_sec()

    def joint_callback(self, msg):
        if 'leg_left_length_joint' in msg.name:
            idx = msg.name.index('leg_left_length_joint')
            self.left_contact = abs(msg.effort[idx]) > self.contact_threshold
        if 'leg_right_length_joint' in msg.name:
            idx = msg.name.index('leg_right_length_joint')
            self.right_contact = abs(msg.effort[idx]) > self.contact_threshold

    def get_kinematics(self):
        try:
            (trans_L, _) = self.listener.lookupTransform(self.base_frame, self.left_foot_frame, rospy.Time(0))
            y_left = np.array(trans_L)
        except: y_left = np.array([0, 0.1, -0.9])

        try:
            (trans_R, _) = self.listener.lookupTransform(self.base_frame, self.right_foot_frame, rospy.Time(0))
            y_right = np.array(trans_R)
        except: y_right = np.array([0, -0.1, -0.9])
            
        return y_left, y_right

    def imu_callback(self, msg):
        if not self.extrinsics_loaded:
            self.load_extrinsics()
            return

        if msg.angular_velocity.x != msg.angular_velocity.x: return 
        
        if not self.filter.is_initialized:
            self.init_buffer.append(msg)
            if len(self.init_buffer) >= self.init_buffer_size:
                self.initialize_filter()
            return

        current_time = msg.header.stamp.to_sec()
        if self.last_imu_time is None:
            self.last_imu_time = current_time
            return
        
        dt = current_time - self.last_imu_time
        self.last_imu_time = current_time
        if dt <= 0 or dt > 1.0: return

        # 1. Rotate Raw IMU data into Base Frame
        w_raw = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        a_raw = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        
        w = np.dot(self.R_imu_base, w_raw)
        a = np.dot(self.R_imu_base, a_raw)
        
        y_left, y_right = self.get_kinematics()

        # 2. Predict
        self.filter.predict(w, a, dt, self.left_contact, self.right_contact, y_left, y_right)
        
        # 3. Correct
        if self.left_contact:
            self.filter.update(y_left, foot_index=1)
        if self.right_contact:
            self.filter.update(y_right, foot_index=0)
        
        self.publish_and_debug(msg.header.stamp)

        if np.isnan(self.filter.X).any():
            rospy.logwarn("Filter Diverged (NaN detected). Resetting...")
            self.filter = InEKF() 
            self.filter.is_initialized = False
            self.init_buffer = []
            self.last_imu_time = None

    def initialize_filter(self):
        acc_sum = np.zeros(3)
        for msg in self.init_buffer:
            a_raw = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
            a = np.dot(self.R_imu_base, a_raw)
            acc_sum += a
        avg_acc = acc_sum / len(self.init_buffer)
        
        last_msg = self.init_buffer[-1]
        q = last_msg.orientation
        
        # R_world_imu
        R_w_i = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        # R_world_base = R_world_imu * R_imu_base^T
        rot_mat = np.dot(R_w_i, self.R_imu_base.T)
        
        g_est = -np.dot(rot_mat, avg_acc)
        self.filter.g_vector = g_est / np.linalg.norm(g_est) * 9.81
        
        self.filter.X[0:3, 0:3] = rot_mat
        self.filter.is_initialized = True
        
        # Initialize feet positions
        y_left, y_right = self.get_kinematics()
        p = self.filter.X[0:3, 4]
        self.filter.X[0:3, 5] = p + np.dot(rot_mat, y_right)
        self.filter.X[0:3, 6] = p + np.dot(rot_mat, y_left)
        
        rospy.loginfo("Filter Initialized!")
        rospy.loginfo("Gravity: %s", str(self.filter.g_vector))

    def get_base_state(self):
        R_base = self.filter.X[0:3, 0:3]
        p_base = self.filter.X[0:3, 4]
        v_base_world = self.filter.X[0:3, 3]
        v_base_body = np.dot(R_base.T, v_base_world)
        return R_base, p_base, v_base_body

    def publish_and_debug(self, stamp):
        R_base, p_base, v_base_body = self.get_base_state()
        
        if self.gt_vel is not None and rospy.Time.now().to_sec() - self.gt_time < 0.5:
            error = np.linalg.norm(v_base_body - self.gt_vel)
            if error > VELOCITY_ERROR_THRESHOLD:
                rospy.logwarn_throttle(1, 
                    "\n>>> VELOCITY MISMATCH DETECTED! <<<\n"
                    "Error: {:.4f} m/s\n"
                    "Est (Base): {}\n"
                    "GT  (Base): {}".format(error, np.round(v_base_body, 3), np.round(self.gt_vel, 3))
                )

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        
        T = np.eye(4)
        T[:3, :3] = R_base
        q = tf.transformations.quaternion_from_matrix(T)
        
        odom.pose.pose.position = Point(p_base[0], p_base[1], p_base[2])
        odom.pose.pose.orientation = Quaternion(q[0], q[1], q[2], q[3])
        odom.twist.twist.linear = Vector3(v_base_body[0], v_base_body[1], v_base_body[2])
        
        self.pub_odom.publish(odom)

if __name__ == '__main__':
    try:
        node = InEKF_ROS()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass