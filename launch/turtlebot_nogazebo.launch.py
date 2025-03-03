"""Launch the TurtleBot3 (Waffle) WITHOUT Gazebo

"""

import os
import xacro

from ament_index_python.packages import get_package_share_directory as pkgdir

from launch                            import LaunchDescription
from launch.actions                    import ExecuteProcess
from launch.actions                    import IncludeLaunchDescription
from launch.actions                    import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions                import Node


#
# Generate the Launch Description
#
def generate_launch_description():

    ######################################################################
    # LOCATE FILES

    # # WORLD: Locate the Gazebo world.  Regular house or alternate world.
    # worldfile = os.path.join(pkgdir('turtlebot3_gazebo'),
    #                          'worlds', 'turtlebot3_house.world')
    # # worldfile = os.path.join(pkgdir('turtlebot3_gazebo'),
    # #                          'worlds', 'turtlebot3_dqn_stage4.world')

    # # MODEL: Locate the Gazebo TurtleBot model: With clean or noisy lidar.
    # # The original is: turtlebot3_gazebo/models/turtlebot3_waffle/model.sdf
    # modelfile = os.path.join(pkgdir('turtlebot'), 'models', 'turtlebot.sdf')
    # cleanlidarmodelfile = os.path.join(pkgdir('turtlebot'), 'models',
    #                                    'turtlebot_cleanlidar.sdf')
    # noisylidarmodelfile = os.path.join(pkgdir('turtlebot'), 'models',
    #                                    'turtlebot_noisylidar.sdf')

    # URDF: Locate and load the TurtleBot URDF, colored orange to stand out.
    # The original is: turtlebot3_description/urdf/turtlebot3_waffle.urdf
    urdffile = os.path.join(pkgdir('turtlebot'), 'urdf', 'turtlebot.urdf')
    with open(urdffile, 'r') as file:
        robot_description = file.read()

    # MAP: Locate the good map
    mapfile = os.path.join(pkgdir('turtlebot'), 'maps/goodmap.yaml')

    # RVIZ: Locate the RVIZ configuration file.
    rvizcfg = os.path.join(pkgdir('turtlebot'), 'rviz/viewturtlebot.rviz')

    # BAGS: Locate the BAG folder.  The first five are in the main house.
    # The alternate recordings use the alternate world (see above).
    #bagfolder = os.path.join(pkgdir('turtlebot'), 'bags', 'singleroom_cleanlaser')
    bagfolder = os.path.join(pkgdir('turtlebot'), 'bags', 'leftside_cleanlaser')
    #bagfolder = os.path.join(pkgdir('turtlebot'), 'bags', 'rightside_cleanlaser')
    #bagfolder = os.path.join(pkgdir('turtlebot'), 'bags', 'singleroom_noisylaser')
    #bagfolder = os.path.join(pkgdir('turtlebot'), 'bags', 'leftside_noisylaser')

    #bagfolder = os.path.join(pkgdir('turtlebot'), 'bags', 'alternate_cleanlaser')
    #bagfolder = os.path.join(pkgdir('turtlebot'), 'bags', 'alternate_noisylaser')


    ######################################################################
    # PREPARE THE LAUNCH ELEMENTS

    ### SIMULATION.
    # # Gazebo Server.  Use the standard launch description.
    # incl_gzserver = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(pkgdir('gazebo_ros'), 'launch', 'gzserver.launch.py')),
    #     launch_arguments={'world': worldfile}.items())

    # # Gazebo Client (Window).  Optional.  Use the standard launch description.
    # incl_gzclient = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(pkgdir('gazebo_ros'), 'launch', 'gzclient.launch.py')))

    # # Spawn the TurtleBot.  This could either be the original, a version
    # # with a perfectly clean lidar scanner, or a version with a noise lidar.
    # node_spawn_turtlebot = Node(
    #     name       = 'spawn_turtlenbot',
    #     package    = 'gazebo_ros',
    #     executable = 'spawn_entity.py',
    #     output     = 'screen',
    #     arguments  = ['-entity', 'waffle', '-file', modelfile,
    #                   '-x', '-2.0', '-y', '1.0', '-z', '0.01'])

    # node_spawn_turtlebot_cleanlidar = Node(
    #     name       = 'spawn_turtlenbot',
    #     package    = 'gazebo_ros',
    #     executable = 'spawn_entity.py',
    #     output     = 'screen',
    #     arguments  = ['-entity', 'waffle', '-file', cleanlidarmodelfile,
    #                   '-x', '-2.0', '-y', '1.0', '-z', '0.01'])

    # node_spawn_turtlebot_noisylidar = Node(
    #     name       = 'spawn_turtlenbot',
    #     package    = 'gazebo_ros',
    #     executable = 'spawn_entity.py',
    #     output     = 'screen',
    #     arguments  = ['-entity', 'waffle', '-file', noisylidarmodelfile,
    #                   '-x', '-2.0', '-y', '1.0', '-z', '0.01'])

    ### SIMULATION ALTERNATIVE
    # ROS Bag Playback.
    cmd_playback = ExecuteProcess(
            cmd    = ['ros2', 'bag', 'play', '--clock', '10', bagfolder],
            output = 'screen')


    ### RVIZ
    # This is configured to show the robot, laser scan, and the map.
    node_rviz = Node(
        name       = 'rviz',
        package    = 'rviz2',
        executable = 'rviz2',
        output     = 'screen',
        arguments  = ['-d', rvizcfg],
        parameters = [{'use_sim_time': True}],
        on_exit    = Shutdown())

    ### URDF PROCESSING
    # Robot State Publisher, to create the TF frames and publish the model.
    node_robot_state_publisher = Node(
        name       = 'robot_state_publisher',
        package    = 'robot_state_publisher',
        executable = 'robot_state_publisher',
        output     = 'screen',
        parameters = [{'robot_description': robot_description},
                      {'use_sim_time':      True}])


    ### LOCALIZATION
    # Select one of the following.  Either assume perfect odometry
    # (which makes the odom frame match the map frame), or assume
    # noisy odometry (which drifts the two frames apart).
    node_perfectlocalization = Node(
        name       = 'localization',
        package    = 'tf2_ros',
        executable = 'static_transform_publisher',
        arguments  = ['--frame-id', 'map', '--child-frame-id', 'odom'])

    node_noisylocalization = Node(
        name       = 'localization',
        package    = 'turtlebot',
        executable = 'noisylocalization',
        output     = 'screen',
        parameters = [{'use_sim_time': True}])


    ### MAPPING
    # The map server publishes an existing map.  It is part of the
    # navigation stack, which requires a lifecycle manager to start.
    node_map_server = Node(
        name       = 'map_server',
        package    = 'nav2_map_server',
        executable = 'map_server',
        output     = 'screen',
        parameters = [{'use_sim_time':  True},
                      {'yaml_filename': mapfile},
                      {'topic_name':    "map"},
                      {'frame_id':      "map"}])

    node_lifecycle = Node(
        name       = 'lifecycle_manager_localization',
        package    = 'nav2_lifecycle_manager',
        executable = 'lifecycle_manager',
        output     = 'screen',
        parameters = [{'use_sim_time': True},
                      {'autostart':    True},
                      {'node_names':   ['map_server']}])

    ### OUR CODE
    # Build the map ourselves.
    node_buildmap = Node(
        name       = 'buildmap',
        package    = 'turtlebot',
        executable = 'buildmap',
        output     = 'screen',
        parameters = [{'use_sim_time': True}])


    ######################################################################
    # COMBINE THE ELEMENTS INTO ONE LIST

    # Return the description, built as a python list.
    return LaunchDescription([

        # SIMULATION: Start the simulation. If you use Gazebo, you
        # have to start the server and ONE of the spawn nodes (with a
        # clean or noisy lidar).  If you do NOT use Gazebo, start the
        # playback using the BAG folder selected at the top.
        #incl_gzserver,
        #node_spawn_turtlebot_cleanlidar,
        #node_spawn_turtlebot_noisylidar,
        cmd_playback,

        # VIEWER: Select a viewer.  RVIZ or Gazebo client.
        #incl_gzclient,
        node_rviz,

        # FRAMES: For anything outside Gazebo, you will need the robot
        # state publisher (to read the URDF and place the frames on
        # the robot), as well as ONE of the localizations (to locate
        # the robot in the map): perfect or noisy.
        node_robot_state_publisher,
        node_perfectlocalization,
        #node_noisylocalization,

        # MAP: Use either the first two lines (together, to show the
        # perfect map).  OR the last line to start your code...
        node_map_server,
        node_lifecycle,
        #node_buildmap,

    ])
