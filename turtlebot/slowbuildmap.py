#!/usr/bin/env python3
#
#   slowbuildmap.py
#
#   Build a map...
#
#   This explores the occupancy grid map.
#
#   This "slow" version explicitly consideres the timestamps, so it
#   can properly connect data even if things are significantly
#   slowed/delayed.
#
#   Node:       /buildmap
#
#   Subscribe:  /scan                   sensor_msgs/Lsaerscan
#   Publish:    /map                    nav_msgs/OccupancyGrid
#
import numpy as np

# ROS Imports
import rclpy

from rclpy.node                 import Node
from rclpy.time                 import Time, Duration
from rclpy.executors            import MultiThreadedExecutor
from rclpy.qos                  import QoSProfile, DurabilityPolicy

from tf2_ros                    import TransformException
from tf2_ros.buffer             import Buffer
from tf2_ros.transform_listener import TransformListener

from geometry_msgs.msg          import Point, Quaternion, Pose
from geometry_msgs.msg          import Transform, TransformStamped
from nav_msgs.msg               import OccupancyGrid
from sensor_msgs.msg            import LaserScan


#
#   Global Definitions
#
WIDTH  = 360
HEIGHT = 240

RESOLUTION = 0.05
ORIGIN_X   = -9.00              # Origin = location of lower-left corner
ORIGIN_Y   = -6.00

LFREE     = 0.0         # FIXME.  Set the log odds ratio of detecting freespace
LOCCUPIED = 0.0         # FIXME.  Set the log odds ratio of detecting occupancy


#
#   Custom Node Class
#
class CustomNode(Node):
    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Create the log-odds-ratio grid.
        self.logoddsratio = np.zeros((HEIGHT, WIDTH))

        # Create a publisher to send the map data.  Note we use a
        # quality of service with durability TRANSIENT_LOCAL, so new
        # subscribers will get the last sent message.  RVIZ and others
        # expect this for map messages.
        quality = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             depth=1)
        self.pub = self.create_publisher(OccupancyGrid, '/map', quality)

        # Instantiate a TF listener. This implicitly fills a local
        # buffer, so we can quickly retrieve the transform information
        # we need.  The buffer is filled via incoming TF message
        # callbacks, so make sure this runs in a seperate thread.
        self.tfBuffer = Buffer()
        TransformListener(self.tfBuffer, self, spin_thread=True)

        # Create a subscriber to the laser scan.
        self.create_subscription(LaserScan, '/scan', self.laserCB, 1)

        # Create a timer for sending the map data.
        self.timer = self.create_timer(1.0, self.sendMap)

    # Shutdown
    def shutdown(self):
        # Destroy the timer and node.
        self.destroy_timer(self.timer)
        self.destroy_node()


    ##################################################################
    # Send the map.  Called from the timer at 1Hz.
    def sendMap(self):
        # Convert the log odds ratio into a probability (0...1).
        # Remember: self.logsoddsratio is a 3460x240 NumPy array,
        # where the values range from -infinity to +infinity.  The
        # probability should also be a 360x240 NumPy array, but with
        # values ranging from 0 to 1, being the probability of a wall.
        FIXME: Convert the log-odds-ratio into a probability.
        probability = ....

        # Perpare the message and send.  Note this converts the
        # probability into percent, sending integers from 0 to 100.
        now  = self.get_clock().now()
        data = (100 * probability).astype(int).flatten().tolist()

        self.map = OccupancyGrid()
        self.map.header.frame_id        = 'map'
        self.map.header.stamp           = now.to_msg()
        self.map.info.map_load_time     = now.to_msg()
        self.map.info.resolution        = RESOLUTION
        self.map.info.width             = WIDTH
        self.map.info.height            = HEIGHT
        self.map.info.origin.position.x = ORIGIN_X
        self.map.info.origin.position.y = ORIGIN_Y
        self.map.data                   = data

        self.pub.publish(self.map)


    ##################################################################
    # Utilities:
    # Set the log odds ratio value
    def set(self, u, v, value):
        # Update only if legal.
        if (u>=0) and (u<WIDTH) and (v>=0) and (v<HEIGHT):
            self.logoddsratio[v,u] = value
        else:
            self.get_logger().warn("Out of bounds (%d, %d)" % (u,v))

    # Adjust the log odds ratio value
    def adjust(self, u, v, delta):
        # Update only if legal.
        if (u>=0) and (u<WIDTH) and (v>=0) and (v<HEIGHT):
            self.logoddsratio[v,u] += delta
        else:
            self.get_logger().warn("Out of bounds (%d, %d)" % (u,v))

    # Return a list of all intermediate (integer) pixel coordinates
    # from (start) to (end) coordinates (which could be non-integer).
    # In classic Python fashion, this excludes the end coordinates.
    def bresenham(self, start, end):
        # Extract the coordinates
        (xs, ys) = start
        (xe, ye) = end

        # Move along ray (excluding endpoint).
        if (np.abs(xe-xs) >= np.abs(ye-ys)):
            return[(u, int(ys + (ye-ys)/(xe-xs) * (u+0.5-xs)))
                   for u in range(int(xs), int(xe), int(np.sign(xe-xs)))]
        else:
            return[(int(xs + (xe-xs)/(ye-ys) * (v+0.5-ys)), v)
                   for v in range(int(ys), int(ye), int(np.sign(ye-ys)))]


    ##################################################################
    # Laserscan CB.  Process the scans.
    def laserCB(self, msg):
        # Grab the transformation between map and laser's scan frames.
        # This checks the time the scan occured and grabs transform
        # from the same time, keeping things consistent.
        tscan = Time().from_msg(msg.header.stamp)
        try:
            tfmsg = self.tfBuffer.lookup_transform(
                'map', msg.header.frame_id, tscan,
                timeout = Duration(seconds=2.0))
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
        rmin     = msg.range_min        # Sensor minimum range to be valid
        rmax     = msg.range_max        # Sensor maximum range to be valid
        ranges   = msg.ranges           # List of ranges for each angle

        thetamin = msg.angle_min        # Min angle (0.0)
        thetamax = msg.angle_max        # Max angle (2pi)
        thetainc = msg.angle_increment  # Delta between angles (2pi/360)
        thetas   = np.arange(thetamin, thetamax, thetainc)

        #############################################################
        # FIXME: PROCESS THE LASER SCAN TO UPDATE THE LOG ODDS RATIO!
        #############################################################


#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the node.
    node = CustomNode('buildmap')

    # Spin the node until interrupted.  To support the TF Listener
    # using another thread, use a multithreaded executor.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except BaseException as ex:
        print("Ending due to exception: %s" % repr(ex))

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
