#!/usr/bin/env python3
#
#   buildmap.py
#
#   Build an occupancy grid map from laser scans.
#
#   Note: The TF chain is now:
#         map → drift → odom → base → laser/scan
#   so the lookup of the transform from 'map' to the laser scan frame
#   will compose the drift and odom corrections.
#
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, DurabilityPolicy

from tf2_ros import Buffer, TransformListener, TransformException

from geometry_msgs.msg import Pose
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan

# Map Dimensions and Resolution
WIDTH       = 360     # number of cells in x
HEIGHT      = 240     # number of cells in y
RESOLUTION  = 0.05    # meters per cell

# Map origin in world coordinates (map.info.origin)
ORIGIN_X    = -9.00
ORIGIN_Y    = -6.00

# Log-odds increments for free/occupied cells
LFREE       = 0.01
LOCCUPIED   = 0.1

class BuildMapNode(Node):
    def __init__(self, name='buildmap'):
        super().__init__(name)
        # Initialize the occupancy grid as log-odds values.
        self.logoddsratio = np.ones((HEIGHT, WIDTH))

        # Publisher for the OccupancyGrid (latched for RViz)
        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.pub = self.create_publisher(OccupancyGrid, '/map', qos)

        # TF listener to transform the laser scan from its frame to the map frame.
        self.tfBuffer = Buffer()
        self.tfListener = TransformListener(self.tfBuffer, self)

        # Track angular and linear velocity from odometry (to skip scans if turning too fast)
        self.angular_velocity = 0.0
        self.linear_speed = 0.0

        # Subscriptions
        self.create_subscription(LaserScan, '/scan', self.laserCB, 1)
        self.create_subscription(Odometry, '/odom', self.odomCB, 1)

        # Timer to publish the map at regular intervals (every 2 seconds)
        self.timer = self.create_timer(2.0, self.sendMap)

    def shutdown(self):
        self.destroy_timer(self.timer)
        self.destroy_node()

    def odomCB(self, msg: Odometry):
        # Update velocities from odometry (used to decide whether to process scans)
        self.angular_velocity = msg.twist.twist.angular.z
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.linear_speed = math.sqrt(vx * vx + vy * vy)

    def sendMap(self):
        # Convert the log odds ratio into a probability (0...1).
        # Remember: self.logsoddsratio is a 3460x240 NumPy array,
        # where the values range from -infinity to +infinity.  The
        # probability should also be a 360x240 NumPy array, but with
        # values ranging from 0 to 1, being the probability of a wall.


        # Perpare the message and send.  Note this converts the
        # probability into percent, sending integers from 0 to 100.
        now  = self.get_clock().now()
        try:
            probability = np.exp(self.logoddsratio) / (1 + np.exp(self.logoddsratio))
            data = (100 * probability).astype(np.int8).flatten().tolist()
        except Exception:
            print("Map error :(")
            return

        grid_msg = OccupancyGrid()
        grid_msg.header.frame_id = 'map'  # The map is the root frame.
        grid_msg.header.stamp = now.to_msg()
        grid_msg.info.map_load_time = now.to_msg()
        grid_msg.info.resolution = RESOLUTION
        grid_msg.info.width = WIDTH
        grid_msg.info.height = HEIGHT
        grid_msg.info.origin.position.x = ORIGIN_X
        grid_msg.info.origin.position.y = ORIGIN_Y
        grid_msg.data = data

        self.pub.publish(grid_msg)

    def update_occupancy_cell(self, x_cell: int, y_cell: int, is_occupied: bool):
        # Adjust the log-odds value for a cell.
        if 0 <= x_cell < WIDTH and 0 <= y_cell < HEIGHT:
            if is_occupied:
                self.logoddsratio[y_cell, x_cell] += LOCCUPIED
            else:
                self.logoddsratio[y_cell, x_cell] -= LFREE

    def bresenham(self, start, end):
        # Bresenham-like ray tracing from start → end in grid coordinates.
        (xs, ys) = start
        (xe, ye) = end
        xs_int = int(xs)
        ys_int = int(ys)
        xe_int = int(xe)
        ye_int = int(ye)

        cells = []
        if abs(xe_int - xs_int) >= abs(ye_int - ys_int):
            step = 1 if xe_int >= xs_int else -1
            slope = (ye - ys) / ((xe - xs) + 1e-9)
            for u in range(xs_int, xe_int, step):
                v = ys + slope * (u + 0.5 - xs)
                cells.append((u, int(v)))
        else:
            step = 1 if ye_int >= ys_int else -1
            slope = (xe - xs) / ((ye - ys) + 1e-9)
            for v in range(ys_int, ye_int, step):
                u = xs + slope * (v + 0.5 - ys)
                cells.append((int(u), v))
        return cells

    def laserCB(self, msg: LaserScan):
        # Build the occupancy grid using the laser scan.
        # Skip updates if the robot is rotating quickly.
        if abs(self.angular_velocity) > 0.1:
            return

        try:
            # Lookup the transform from 'map' to the scan frame.
            tfmsg = self.tfBuffer.lookup_transform('map', msg.header.frame_id, Time())
        except TransformException as ex:
            self.get_logger().warn(f"Unable to get transform: {ex}")
            return

        # Get the scanner’s pose in the map frame.
        xc = tfmsg.transform.translation.x
        yc = tfmsg.transform.translation.y
        thetac = 2.0 * math.atan2(tfmsg.transform.rotation.z, tfmsg.transform.rotation.w)

        # Convert to grid coordinates.
        u0 = (xc - ORIGIN_X) / RESOLUTION
        v0 = (yc - ORIGIN_Y) / RESOLUTION

        rmin = msg.range_min
        rmax = msg.range_max
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = msg.ranges

        for angle, rng in zip(angles, ranges):
            direction = thetac + angle
            if rmin < rng < rmax:
                dist_in_cells = rng / RESOLUTION
                u_f = u0 + dist_in_cells * math.cos(direction)
                v_f = v0 + dist_in_cells * math.sin(direction)
                # Mark free cells along the ray.
                line_cells = self.bresenham((u0, v0), (u_f, v_f))
                for (uu, vv) in line_cells:
                    self.update_occupancy_cell(uu, vv, is_occupied=False)
                # Mark the endpoint as occupied.
                self.update_occupancy_cell(int(u_f), int(v_f), True)
            elif rng >= rmax:
                dist_in_cells = rmax / RESOLUTION
                u_f = u0 + dist_in_cells * math.cos(direction)
                v_f = v0 + dist_in_cells * math.sin(direction)
                line_cells = self.bresenham((u0, v0), (u_f, v_f))
                for (uu, vv) in line_cells:
                    self.update_occupancy_cell(uu, vv, is_occupied=False)

def main(args=None):
    rclpy.init(args=args)
    node = BuildMapNode()
    rclpy.spin(node)
    node.shutdown()
    rclpy.shutdown()

if __name__ == '__main__':
    main()