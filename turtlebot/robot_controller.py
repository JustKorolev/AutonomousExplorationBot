import time
import numpy as np

def wrapto180(angle):
    """Normalize an angle to the range [-π, π]."""
    return angle - 2 * np.pi * round(angle / (2 * np.pi))

class RobotController:
    def __init__(self, Kp_ang=0.5, Kd_ang=-0.2, Kp_lin=1, Kd_lin=0.3, goal_tolerance=0.05, alpha=0.2):
        self.pos = np.zeros(2)
        self.orientation = 0
        self.path = None
        self.Kp_ang = Kp_ang
        self.Kd_ang = Kd_ang
        self.Kp_lin = Kp_lin
        self.Kd_lin = Kd_lin
        self.goal_tolerance = goal_tolerance
        self.prev_angle = 0.0
        self.prev_distance = 0.0
        self.filtered_angle_error = 0.0  # For the EMA filter
        self.alpha = alpha  # Smoothing factor for the EMA filter
        self.current_index = 0
        self.prev_time = 0
        self.timeout = 15

    def update_path(self, path):
        self.path = path

    def reset(self):
        """Reset the controller's state variables."""
        self.pos = np.zeros(2)
        self.orientation = 0
        self.path = None
        self.prev_angle = 0.0
        self.prev_distance = 0.0
        self.filtered_angle_error = 0.0
        self.current_index = 0


    def run(self, node, get_odom, send_velocity, is_colliding, spin):
        self.prev_time = time.time()
        while self.current_index < len(self.path):
            spin(node)  # Spin the node that is running this
            target = np.array(self.path[self.current_index])
            position, theta_rad = get_odom()

            # Calculate direction and distance to the target
            direction = target - position
            distance = np.linalg.norm(direction)
            target_rad = wrapto180(np.atan2(direction[1], direction[0]))
            angle_error = wrapto180(target_rad - theta_rad)

            # Apply exponential moving average (EMA) to filter angle_error
            self.filtered_angle_error = (
                self.alpha * self.filtered_angle_error + (1 - self.alpha) * angle_error
            )


            # Check if the waypoint is reached
            if distance < self.goal_tolerance:
                self.current_index += 1
                self.prev_angle = 0.0
                self.filtered_angle_error = 0.0
                self.prev_distance = 0.0  # Reset the previous distance
                continue

            # Prevent running into newly found obstacles
            normalized_direction = direction / distance
            if is_colliding(position, normalized_direction):
                print("OBSTACLE IN PATH - ABORTING")
                break


            # PD control for angular velocity
            now = time.time()
            dt = now - self.prev_time
            angular_velocity = (
                self.Kp_ang * angle_error
                + self.Kd_ang * (theta_rad - self.prev_angle) / dt
            )
            self.prev_angle = theta_rad

            # PD control for linear velocity
            distance_derivative = (distance - self.prev_distance) / dt
            linear_velocity = self.Kp_lin * distance + self.Kd_lin * distance_derivative
            self.prev_distance = distance

            # Only drive forward if facing the right direction
            if abs(self.filtered_angle_error) >= 0.1:
                linear_velocity = 0.0

            # Send the velocity command
            send_velocity(linear_velocity, angular_velocity)
            self.prev_time = now

        send_velocity(0.0, 0.0) # Stop after path execution