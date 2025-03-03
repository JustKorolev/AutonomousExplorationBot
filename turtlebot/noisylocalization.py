#!/usr/bin/env python3
#
#   noisylocalization.py
#
#   Simulate noisy odometry or drifting localization.
#
#   Node:       /localization
#
#   Subscribe:  /odom                   sensor_msgs/Lsaerscan
#   Publish:    -nothing-
#
#   TF Broadcast:  odom frame in map fram
#
import numpy as np

from math                       import pi, sin, cos, atan2, sqrt
from scipy.spatial      import KDTree

# ROS Imports
import rclpy

from rclpy.node                 import Node
from rclpy.time                 import Time

from tf2_ros                    import TransformBroadcaster
from tf2_ros                    import TransformException
from tf2_ros.buffer             import Buffer
from tf2_ros.transform_listener import TransformListener
from rclpy.qos                  import QoSProfile, DurabilityPolicy

from geometry_msgs.msg          import Point, Quaternion, Pose
from geometry_msgs.msg          import Transform, TransformStamped
from nav_msgs.msg               import Odometry
from nav_msgs.msg               import OccupancyGrid
from sensor_msgs.msg            import LaserScan


#
#   Global Definitions
#
R = 0.3                 # Radius to convert angle to distance.

NOISE = 0.12            # Noise fraction

OBSTACLE_THRES = 90


#
#   Angle Wrapping
#
def wrapto90(angle):
    return angle -   pi * round(angle/(  pi))
def wrapto180(angle):
    return angle - 2*pi * round(angle/(2*pi))


#
#   Custom Node Class
#
class CustomNode(Node):
    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Set the initial drift value.
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Set the initial odometry reading.
        self.odom = None

        # Set the initial map to empty
        self.map = OccupancyGrid()
        self.scan = LaserScan()

        # Initialize the transform broadcaster
        self.tfBroadcaster = TransformBroadcaster(self)

        quality = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             depth=1)
        # Create a subscriber to the odometry topic.
        self.create_subscription(Odometry, '/odom', self.odomCB, 1)
        self.create_subscription(LaserScan, '/scan', self.updateScan, 1)
        self.create_subscription(OccupancyGrid, '/map', self.updateMap, quality)

        # Instantiate a TF listener.
        self.tfBuffer   = Buffer()
        self.tfListener = TransformListener(self.tfBuffer, self)

    # Shutdown
    def shutdown(self):
        # No particular cleanup, just shut down the node.
        self.destroy_node()

    def updateMap(self, msg):
        self.map = msg

    def updateScan(self, msg):
        self.scan = msg

    def correct_odometry(self):
        try:
            tfmsg = self.tfBuffer.lookup_transform(
                'map', self.scan.header.frame_id, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().warn("Unable to get transform: %s" % (ex,))
            return

        # Extract the laser scanner's position and orientation.
        xc     = tfmsg.transform.translation.x
        yc     = tfmsg.transform.translation.y
        thetac = 2 * np.arctan2(tfmsg.transform.rotation.z,
                                tfmsg.transform.rotation.w)

        # Grab the rays: each ray's range and angle relative to the
        # turtlebot's position and orientation.
        rmin     = self.scan.range_min        # Sensor minimum range to be valid
        rmax     = self.scan.range_max        # Sensor maximum range to be valid
        ranges   = self.scan.ranges           # List of ranges for each angle

        thetamin = self.scan.angle_min        # Min angle (0.0)
        thetamax = self.scan.angle_max        # Max angle (2pi)
        thetainc = self.scan.angle_increment  # Delta between angles (2pi/360)
        thetas   = np.arange(thetamin, thetamax, thetainc)

        ORIGIN_X = self.map.info.origin.position.x
        ORIGIN_Y = self.map.info.origin.position.y
        RESOLUTION = self.map.info.resolution
        HEIGHT = self.map.info.height
        WIDTH = self.map.info.width
        if RESOLUTION == 0:
            return

        scan_points = []
        u, v = (xc - ORIGIN_X) / RESOLUTION, (yc - ORIGIN_Y) / RESOLUTION

        for theta, _range in zip(thetas, ranges):
            direction = theta + thetac
            if rmin < _range < rmax:
                dist = _range / RESOLUTION
                u_f, v_f = u + dist*np.cos(direction), v + dist*np.sin(direction)
                scan_points.append((u_f, v_f))

        map_points = []
        map_data = np.array(self.map.data).reshape((WIDTH, HEIGHT))
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if map_data[x, y] > OBSTACLE_THRES:
                    map_points.append((x, y))
    

        try:
            kdtree  = KDTree(map_points)
            radius = 1
            near_map_indices = set(np.hstack(kdtree.query_ball_point(scan_points, r=radius)))
            # near_map_indices = [kdtree.query(scan_point, k=1)[1] for scan_point in scan_points]
            near_map_points = [map_points[int(index)] for index in near_map_indices]
            N = min(len(near_map_points), len(scan_points))
            near_map_points = near_map_points[:N]
            scan_points = scan_points[:N]

            P = np.array(near_map_points)
            Q = np.array(scan_points)

            # Compute centroids
            p_centroid = np.mean(P, axis=0)
            q_centroid = np.mean(Q, axis=0)

            # Compute offset arrays
            P_offset = P - p_centroid
            Q_offset = Q - q_centroid

            # Compute Covariance Matrix
            H = P_offset.T @ Q_offset

            # Calculate SVD
            U, _, Vh = np.linalg.svd(H)
            d = np.linalg.det(U @ Vh)
            m = np.eye(2)
            m[1, 1] = d
            R = U @ m @ Vh
            t = p_centroid - R @ q_centroid
            print(p_centroid - q_centroid)
            print("^^^^^")
        except Exception as e:
            print(e)
            return

        if N > 50:
            pos_prev = np.array([self.x, self.y])
            pos_new = (R @ pos_prev.T).T + t
            self.x, self.y = pos_new
            print("HERE")
            rot_vec = np.array([cos(self.theta), sin(self.theta)])
            new_rot_vec = R @ rot_vec
            self.theta = np.arctan2(new_rot_vec[1], new_rot_vec[0])
            # print(pos_prev)
            # print(pos_new)
            # new_theta = R.T @ self.theta





    # Odometry CB.  See how far we've moved and drift accordingly.
    def odomCB(self, msg):
        # Check if we have an old value.
        if self.odom is None:
            self.odom = msg.pose.pose
            return

        # Grab the new odometry reading.
        x1 = msg.pose.pose.position.x
        y1 = msg.pose.pose.position.y
        t1 = 2 * atan2(msg.pose.pose.orientation.z,
                       msg.pose.pose.orientation.w)

        # Grab the old odometry reading.
        x0 = self.odom.position.x
        y0 = self.odom.position.y
        t0 = 2 * atan2(self.odom.orientation.z,
                       self.odom.orientation.w)

        # Save the new reading.
        self.odom = msg.pose.pose

        # Compute the magnitude of the difference.
        d = sqrt((x1-x0)**2 + (y1-y0)**2 + (R*wrapto180(t1-t0))**2)

        # Drift we want on the new position.
        if True:
            dx = np.random.uniform(-d  *NOISE, d  *NOISE)
            dy = np.random.uniform(-d  *NOISE, d  *NOISE)
            dt = np.random.uniform(-d/R*NOISE, d/R*NOISE)
        else:
            dx = -0.1 * (x1-x0)
            dy = -0.1 * (y1-y0)
            dt = -0.1 * wrapto180(t1-t0)

        # Matching drift to the odometry frame.
        dx += (1-cos(dt))*x1 + sin(dt)*y1
        dy += (1-cos(dt))*y1 - sin(dt)*x1

        # Update the odometry frame.
        self.x     += cos(self.theta) * dx - sin(self.theta) * dy
        self.y     += sin(self.theta) * dx + cos(self.theta) * dy
        self.theta += dt

        # Update odometry based on map vs lidar
        self.correct_odometry()

        # Broadcast the new drift.
        tfmsg = TransformStamped()

        tfmsg.header.stamp            = msg.header.stamp
        tfmsg.header.frame_id         = 'map'
        tfmsg.child_frame_id          = 'odom'
        tfmsg.transform.translation.x = self.x
        tfmsg.transform.translation.y = self.y
        tfmsg.transform.rotation.z    = sin(self.theta/2)
        tfmsg.transform.rotation.w    = cos(self.theta/2)

        self.tfBroadcaster.sendTransform(tfmsg)

        #print('Sent transform (%6.3f, %6.3f, %6.3f)' %
        #      (self.x, self.y, self.theta))


#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the node.
    node = CustomNode('odometry')

    # Spin, until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
