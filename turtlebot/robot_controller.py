import time
import numpy as np

def wrapto180(angle):
    return angle - 2*np.pi * round(angle/(2*np.pi))

class RobotController:
    def __init__(self, Kp_ang=1.2, Kd_ang=0.001, Kp_lin=0.1, goal_tolerance=0.1):
        self.pos = np.zeros(2)
        self.orientation = 0
        self.path = None
        self.Kp_ang = Kp_ang
        self.Kd_ang = Kd_ang
        self.Kp_lin = Kp_lin
        self.goal_tolerance = goal_tolerance
        self.prev_angle_error = 0.0
        self.current_index = 0
        self.prev = time.time()

    def update_path(self, path):
        self.path = path

    def run(self, node, get_odom, send_velocity, spin):
        while self.current_index < len(self.path):
            spin(node) # spin the node that is running this
            target = np.array(self.path[self.current_index])
            position, theta = get_odom()
            theta += 0.25

            direction = target - position
            distance = np.linalg.norm(direction)
            target_angle = (np.arctan2(direction[1], direction[0]))
            angle_error = target_angle - theta
            # print(f"theta: {theta}")
            # print(f"target: {target_angle}")
            # print(f"error: {angle_error}")


            if distance < self.goal_tolerance:
                print(f"Reached waypoint {self.current_index}: {target}")
                self.current_index += 1
                self.prev_angle_error = 0.0
                continue

            # PD control for angular velocity
            now = time.time()
            dt = now - self.prev
            angular_velocity = -self.Kp_ang * angle_error + self.Kd_ang * (angle_error - self.prev_angle_error) / dt
            self.prev_angle_error = angle_error

            # Only drive forward if facing the right direction
            if abs(angle_error) < 0.1:
                linear_velocity = self.Kp_lin * distance
            else:
                linear_velocity = 0.0

            send_velocity(linear_velocity, angular_velocity)
            self.prev = now