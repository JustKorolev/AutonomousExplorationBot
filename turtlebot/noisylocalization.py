#!/usr/bin/env python3

import numpy as np
import math
from math import pi, sin, cos, atan2, sqrt
from scipy.spatial import KDTree

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import Point, TransformStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import TransformBroadcaster, TransformException, Buffer, TransformListener
# import logging

# ----------------------------------------------------
# User-configurable parameters
# ----------------------------------------------------
OBSTACLE_THRES         = 85    # Occupancy threshold for obstacles.
TURN_THRESHOLD         = 0.1   # Skip scan update if turning too fast.
TIME_OFFSET            = 0.05  # (Not used now, since we use latest transform)
CAN_TRANSFORM_TIMEOUT  = 0.3   # Seconds.

UPDATE_FACTOR   = 0.05         # Fraction to integrate per scan.
MAX_MEAN_ERROR  = 0.5          # Maximum allowed mean alignment error (m).
MAX_UPDATE_NORM = 0.1          # Maximum allowed translation update (m).
EXTRAPOLATION_THRESHOLD = 0.3  # Maximum allowed transform time difference.

# Noise standard deviations (tweak these values)
NOISE_STD_X     = 0.05        # meters
NOISE_STD_Y     = 0.05        # meters
NOISE_STD_THETA = 0.05        # radians

def wrapto180(angle):
    return angle - 2 * pi * round(angle/(2*pi))

class CustomNode(Node):
    def __init__(self, name):
        super().__init__(name)
        # Adjust logfile path as needed.
        # logging.basicConfig(
        #     filename='/home/baaqerfarhat/robotws/src/turtlebot/turtlebot/logfile.log',
        #     level=logging.INFO,
        #     format='%(asctime)s %(message)s'
        # )
        # self.logger = logging.getLogger(__name__)

        # ---- Fixed drift→odom error simulation ----
        self.fixed_x_error = 0.0
        self.fixed_theta_error = 0.0
        self.fixed_y_error = 0.0

        self.x     = self.fixed_x_error
        self.y     = self.fixed_y_error
        self.theta = self.fixed_theta_error

        # Map→drift correction (initialize to identity).
        self.dx_map = 0.0
        self.dy_map = 0.0
        self.dt_map = 0.0

        self.map = OccupancyGrid()
        self.latest_scan = None
        self.angular_velocity = 0.0

        # TF Broadcaster
        self.tfBroadcaster = TransformBroadcaster(self)
        qos_transient = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization_marker_array', 1)

        # Subscriptions
        self.create_subscription(Odometry, '/odom', self.odomCB, 1)
        self.create_subscription(LaserScan, '/scan', self.scanCB, 1)
        self.create_subscription(OccupancyGrid, '/map', self.updateMap, qos_transient)

        # TF Buffer/Listener
        self.tfBuffer = Buffer()
        self.tfListener = TransformListener(self.tfBuffer, self)

    def shutdown(self):
        self.destroy_node()

    def updateMap(self, msg):
        self.map = msg
        # self.get_logger().info("Map updated.")

    def odomCB(self, msg: Odometry):
        self.angular_velocity = msg.twist.twist.angular.z
        self.x = self.fixed_x_error
        self.y = self.fixed_y_error
        self.theta = self.fixed_theta_error

        # self.get_logger().info(
        #     f"Fixed Drift→odom: x={self.x:.2f}, y={self.y:.2f}, theta={self.theta:.2f}"
        # )

        tfmsg = TransformStamped()
        tfmsg.header.stamp = msg.header.stamp
        tfmsg.header.frame_id = 'drift'
        tfmsg.child_frame_id  = 'odom'
        tfmsg.transform.translation.x = self.x
        tfmsg.transform.translation.y = self.y
        tfmsg.transform.rotation.z = math.sin(self.theta/2.0)
        tfmsg.transform.rotation.w = math.cos(self.theta/2.0)
        self.tfBroadcaster.sendTransform(tfmsg)
        self.broadcastMapCorrection()

    def transform_point(self, point, dx, dy, dtheta):
        # Assuming point is a 2D coordinate (x, y)
        x, y = point
        # Apply the rotation (dtheta) and translation (dx, dy)
        x_new = cos(dtheta) * x - sin(dtheta) * y + dx
        y_new = sin(dtheta) * x + cos(dtheta) * y + dy
        return (x_new, y_new)

    def publish_line_marker(self, point_pairs):
        markers = []
        now = self.get_clock().now().to_msg()
        for i, (ptA, ptB) in enumerate(point_pairs):
            # Here you might apply a condition to only show markers if the distance is below 0.2
            distance = math.sqrt((ptA[0] - ptB[0])**2 + (ptA[1] - ptB[1])**2)
            if distance > 0.2:
                continue

            mk = Marker()
            mk.header.frame_id = "map"
            mk.type = Marker.LINE_STRIP
            mk.action = Marker.ADD
            mk.id = i
            mk.ns = "line_markers"
            mk.header.stamp = now
            mk.lifetime = Duration(seconds=0).to_msg()  # persistent
            mk.scale.x = 0.1  # line width
            mk.color.r = 0.0
            mk.color.g = 0.7
            mk.color.b = 0.5
            mk.color.a = 1.0
            pA = Point(x=float(ptA[0]), y=float(ptA[1]))
            pB = Point(x=float(ptB[0]), y=float(ptB[1]))
            mk.points.append(pA)
            mk.points.append(pB)
            markers.append(mk)
        self.marker_pub.publish(MarkerArray(markers=markers))


    def computeAlignmentLS(self, scan: LaserScan):
        # Instead of using the scan timestamp, use the latest available transform.
        try:
            tfmsg = self.tfBuffer.lookup_transform(
                'drift', scan.header.frame_id,
                Time(),  # Use the current time to get the latest transform.
                timeout=Duration(seconds=CAN_TRANSFORM_TIMEOUT))
        except TransformException as ex:
            # self.get_logger().warn(f"Unable to get drift->scan transform using latest time: {ex}")
            return None

        # Optionally, you can check the time difference if needed.
        # (If you do, make sure it doesn't block or cause extrapolation issues.)

        # --- Add noise to the sensor reading ---
        sensor_x = tfmsg.transform.translation.x + np.random.normal(0, NOISE_STD_X)
        sensor_y = tfmsg.transform.translation.y + np.random.normal(0, NOISE_STD_Y)
        sensor_theta = (2.0 * math.atan2(tfmsg.transform.rotation.z, tfmsg.transform.rotation.w)
                        + np.random.normal(0, NOISE_STD_THETA))

        # Compute scan points in the drift frame.
        rmin = scan.range_min
        rmax = scan.range_max
        angles = np.arange(scan.angle_min, scan.angle_max, scan.angle_increment)
        scan_pts_drift = []
        for ang, rng in zip(angles, scan.ranges):
            if rmin < rng < rmax:
                effective_ang = sensor_theta + ang
                x_d = sensor_x + rng * math.cos(effective_ang)
                y_d = sensor_y + rng * math.sin(effective_ang)
                scan_pts_drift.append((x_d, y_d))
        if len(scan_pts_drift) < 20:
            return None

        Q = np.array(scan_pts_drift)

        # Extract obstacle points from the occupancy grid.
        H = self.map.info.height
        W = self.map.info.width
        ORIGIN_X = self.map.info.origin.position.x
        ORIGIN_Y = self.map.info.origin.position.y
        RES = self.map.info.resolution
        grid = np.array(self.map.data).reshape((H, W))
        map_pts = []
        for row in range(H):
            for col in range(W):
                if grid[row, col] > OBSTACLE_THRES:
                    map_pts.append((col, row))
        if len(map_pts) < 20:
            return None

        def unscale(arr):
            arr = np.array(arr, dtype=float)
            arr[:, 0] = arr[:, 0] * RES + ORIGIN_X
            arr[:, 1] = arr[:, 1] * RES + ORIGIN_Y
            return arr

        P_all = unscale(map_pts)

        kdtree = KDTree(P_all)
        idxs = [kdtree.query(q, k=1)[1] for q in Q]
        P = np.array([P_all[i] for i in idxs])

        diffs = np.linalg.norm(P - Q, axis=1)
        mean_diff = np.mean(diffs)
        # self.logger.info(f"Differences before correction: mean={mean_diff:.2f}")
        self.publish_line_marker(zip(P, Q))

        if len(P) < 20 or mean_diff > MAX_MEAN_ERROR:
            # self.get_logger().warn(f"Alignment error {mean_diff:.2f} exceeds threshold or too few points. Skipping update.")
            return None

        N = len(P)
        Rx = np.mean(Q[:,0])
        Ry = np.mean(Q[:,1])
        Px = np.mean(P[:,0])
        Py = np.mean(P[:,1])
        sum_RR = np.sum(Q[:,0]**2 + Q[:,1]**2)
        sum_RP = np.sum(Q[:,0]*P[:,1] - Q[:,1]*P[:,0])
        RR = sum_RR / N
        RP = sum_RP / N

        denominator = RR - (Rx**2 + Ry**2)
        if abs(denominator) < 1e-9:
            # self.get_logger().warn("Denominator too small; skipping update.")
            return None

        dth_update = (RP - (Rx*Py - Ry*Px)) / denominator
        dx_update = (Px - Rx) + Ry * dth_update
        dy_update = (Py - Ry) - Rx * dth_update

        dx_update = -dx_update
        dy_update = -dy_update
        dth_update = -dth_update

        update_norm = math.sqrt(dx_update**2 + dy_update**2)
        if update_norm > MAX_UPDATE_NORM:
            scale = MAX_UPDATE_NORM / update_norm
            dx_update *= scale
            dy_update *= scale

        return (dx_update, dy_update, dth_update)

    def broadcastMapCorrection(self):
        tf_corr = TransformStamped()
        tf_corr.header.stamp = self.get_clock().now().to_msg()
        tf_corr.header.frame_id = 'map'
        tf_corr.child_frame_id  = 'drift'
        tf_corr.transform.translation.x = self.dx_map
        tf_corr.transform.translation.y = self.dy_map
        tf_corr.transform.rotation.z = math.sin(self.dt_map/2.0)
        tf_corr.transform.rotation.w = math.cos(self.dt_map/2.0)
        self.tfBroadcaster.sendTransform(tf_corr)
        # self.get_logger().info(
        #     f"Drift Correction (map→drift): dx_map={self.dx_map:.2f}, dy_map={self.dy_map:.2f}, dt_map={self.dt_map:.2f}"
        # )


    def verify_correction(self):
        # Compute a combined correction magnitude (you can adjust this as needed)
        corr_norm = math.sqrt(self.dx_map**2 + self.dy_map**2 + self.dt_map**2)
        # if corr_norm > 0.001:
        #     self.get_logger().info(f"CORRECTED BY {corr_norm:.2f} SUCCESS")
        # else:
        #     self.get_logger().info("No significant correction applied.")

    def scanCB(self, scan: LaserScan):
        if abs(self.angular_velocity) >= TURN_THRESHOLD:
            return

        update = self.computeAlignmentLS(scan)
        if update is None:
            return
        dx_u, dy_u, dth_u = update

        dx_u  *= UPDATE_FACTOR
        dy_u  *= UPDATE_FACTOR
        dth_u *= UPDATE_FACTOR

        new_dx = self.dx_map + cos(self.dt_map)*dx_u - sin(self.dt_map)*dy_u
        new_dy = self.dy_map + sin(self.dt_map)*dx_u + cos(self.dt_map)*dy_u
        new_dt = self.dt_map + dth_u

        self.dx_map, self.dy_map, self.dt_map = new_dx, new_dy, new_dt

        # self.get_logger().info(
        #     f"[LS Correction] dX={dx_u/UPDATE_FACTOR:.2f}, dY={dy_u/UPDATE_FACTOR:.2f}, dTh={dth_u/UPDATE_FACTOR:.2f}.  "
        #     f"Accumulated: (dx_map={self.dx_map:.2f}, dy_map={self.dy_map:.2f}, dt_map={self.dt_map:.2f})"
        # )
        self.broadcastMapCorrection()
        # Verify the correction and print a message if significant.
        self.verify_correction()

def main(args=None):
    rclpy.init(args=args)
    node = CustomNode('localization_ls')
    rclpy.spin(node)
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
