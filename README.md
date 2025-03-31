# Autonomous Exploration and Localization Correction (via frontier-based RRT & least-squares alignment)

## Contributors
Andrey Korolev, Baaqer Farhat

## Overview
This project demonstrates the use of **SLAM** and **frontier-based exploration** to explore unknown environments autonomously.
Localization correction is implemented using a **least-squares alignment method** to correct the robot's pose relative to a **log-odds occupancy grid**.
The frontier exploration consists of detecting valid frontier cells, selecting a goal based on a chosen cost function, and running **RRT** to find a path in real time.
The combination of these two features allows for a robust and rapid exploration of large maps. Individual parameters can be tuned to improve performance based on prior known facts about the environment.

![](assets/gifs/small_map.gif)

## Key Features

- **Frontier-Based Exploration**: Identifies unexplored regions using occupancy grid analysis and selects high-value goals based on a heuristic function.
- **Path Planning with RRT**: Uses a modified RRT algorithm with goal biasing to generate paths quickly while prioritizing exploration over path optimality.
- **Collision Avoidance**: Dynamically updates frontier weights and path selection based on detected obstacles.
- **PD Controller for Motion Execution**: Controls linear and angular velocity to follow planned paths efficiently.

### RRT Path Planning
- The algorithm selects random nodes with a **40% bias toward the goal**.
- It expands the tree by **taking steps of size 4** towards the nearest valid node.
- The final path is **post-processed** to minimize redundant waypoints.

### Collision Avoidance
- The system inflates obstacles using a **collision clearance radius of 9**.
- If a potential collision is detected, the system updates frontier weights to discourage selecting unreachable goals.

## Execution Flow
1. The system receives an occupancy grid map from ROS.
2. It identifies frontiers and selects a high-value goal.
3. RRT generates a path to the goal.
4. The path is optimized and executed using PD control.
5. The system continuously updates goals and replans as necessary.

