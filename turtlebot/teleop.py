#!/usr/bin/env python3
#
#   teleop.py
#
#   Continually (at 10Hz!) send the a velocity command.
#
#   Node:       /teleop
#
#   Subscribe:  -nothing-
#   Publish:    /cmd_vel        geometry_msgs/Twist
#
import curses
import numpy as np
import sys

# ROS Imports
import rclpy

from rclpy.node                 import Node
from rclpy.time                 import Time

from geometry_msgs.msg          import Twist


#
#   Global Definitions
#
VNOM = 0.25
WNOM = 0.5             # This gives a turning radius of v/w = 0.5m


#
#   Custom Node Class
#
class CustomNode(Node):
    # Initialization.
    def __init__(self, name, vnom, wnom):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Save the parameters.
        self.vnom = vnom
        self.wnom = wnom

        # Create a publisher to send twist commands.
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Initialize the (repeating) message data.
        self.msg = Twist()
        self.msg.linear.x = 0.0
        self.msg.linear.y = 0.0
        self.msg.linear.z = 0.0
        self.msg.angular.x = 0.0
        self.msg.angular.y = 0.0
        self.msg.angular.z = 0.0

        # Create a fixed rate to control the speed of sending commands.
        rate    = 10.0
        self.dt = 1/rate
        self.rate = self.create_rate(rate)

        # Report.
        self.get_logger().info("Teleop sending every %f sec..." % self.dt)
        self.get_logger().info("Nominal fwd = %6.3fm/s, spin = %6.3frad/sec"
                               % (self.vnom, self.wnom))

    # Shutdown
    def shutdown(self):
        # Destroy the rate and node.
        self.destroy_rate(self.rate)
        self.destroy_node()


    # Run the terminal input loop, send the commands, and ROS spin.
    def loop(self, screen):
        # Make the getchar() non-blocking and otherwise initialize.
        curses.curs_set(0)
        curses.flushinp()
        screen.nodelay(True)
        screen.erase()
        screen.addstr(0,  0, "To move press and hold:    t y u")
        screen.addstr(1,  0, "(h = halt immediately)     g h j")
        screen.addstr(2,  0, "                           b n m")
        screen.addstr(3,  0, "Commands last 0.5sec")
        screen.addstr(4,  0, "Add SHIFT for high speed (2x)")
        screen.addstr(5,  0, "Add CTRL  for slow speed (0.25x)")
        screen.addstr(6,  0, "Hit q to quit")
        screen.addstr(10, 0, "Nominal fwd = %6.3fm/s, spin = %6.3frad/sec" %
                      (self.vnom, self.wnom))

        # Set up the map from keycodes (0..127) to velocities.
        nom  = [(0.0, 0.0)] * 0x20
        nom[ord('y') & 0x1F] = ( self.vnom,        0.0)  # y = forward
        nom[ord('n') & 0x1F] = (-self.vnom,        0.0)  # n = backward
        nom[ord('g') & 0x1F] = (       0.0,  self.wnom)  # g = left spin
        nom[ord('j') & 0x1F] = (       0.0, -self.wnom)  # j = right spin
        nom[ord('t') & 0x1F] = ( self.vnom,  self.wnom)  # t = foward + left
        nom[ord('u') & 0x1F] = ( self.vnom, -self.wnom)  # t = foward + right
        nom[ord('b') & 0x1F] = (-self.vnom,  self.wnom)  # b = backward + left
        nom[ord('m') & 0x1F] = (-self.vnom, -self.wnom)  # m = backward + right

        map  = []
        map += [(0.25*vx, 0.25*wz) for (vx, wz) in nom]  # Control   = slow
        map += [(0.0 *vx, 0.0 *wz) for (vx, wz) in nom]  # Numbers   = nothing
        map += [(2.0 *vx, 2.0 *wz) for (vx, wz) in nom]  # Uppercase = fast
        map += [(1.0 *vx, 1.0 *wz) for (vx, wz) in nom]  # Lowercase = nominal

        # Initialize the velocity and remaining active time.
        key = 0
        vel = (0.0, 0.0)
        Tactive = 0.0

        # Run the loop until shutdown.
        while rclpy.ok():
            # Reduce the active time and stop if necessary.
            Tactive -= self.dt
            if Tactive <= 0.0:
                vel = (0.0, 0.0)
                Tactive = 0.0

            # Process all pending keys.
            while True:
                keycode = screen.getch()
                if keycode == -1:
                    break

                # Reset the active time.
                key     = keycode
                Tactive = 0.5

                # Change the velocity based on the key.
                if not ((keycode >= 0) and (keycode < 0x80)):
                    vel = (0.0, 0.0)
                elif (keycode & 0x1F) == (ord('q') & 0x1F):
                    return
                else:
                    vel = map[keycode]

            # Report.
            if (key >= 0x20) and (key <= 0x80):
                s = "'" + str(chr(key)) + "'"
            else:
                s = "%d" % key
            screen.addstr(8, 0, "Last pressed 0x%02x = %s" % (key, s))
            screen.clrtoeol()
            screen.addstr(11, 0,
                          "Sending fwd = %6.3fm/s, spin = %6.3frad/sec" % vel)

            # Update the message and publish.
            self.msg.linear.x  = vel[0]
            self.msg.angular.z = vel[1]
            self.pub.publish(self.msg)

            # Spin once to process other items.
            rclpy.spin_once(self)

            # Wait for the next turn.
            self.rate.sleep()


#
#   Main Code
#
def main(args=None):
    # Pull out the nominal forward/spin speeds from the non-ROS parameters.
    if (len(sys.argv) > 3):
        print("Usage: teleop.py forward_speed spin_speed")
        print("GOOD DEFAULTS: teleop.py %3.1f %3.1f" % (VNOM, WNOM))
        return
    elif (len(sys.argv) < 3):
        print("Usage: teleop.py forward_speed spin_speed")
        print("Using default values...")
        vnom = VNOM
        wnom = WNOM
    else:
        vnom = float(sys.argv[1])
        wnom = float(sys.argv[2])

    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the node.
    node = CustomNode('teleop', vnom, wnom)

    # Run the terminal input loop, which spins the ROS items.
    try:
        curses.wrapper(node.loop)
    except KeyboardInterrupt:
        pass

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
