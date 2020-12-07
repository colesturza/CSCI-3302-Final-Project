"""maze_mapper controller."""

import math
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from controller import Robot, Motor, DistanceSensor

LIDAR_SENSOR_MAX_RANGE = 3.  # Meters
LIDAR_ANGLE_BINS = 21  # 21 Bins to cover the angular range of the lidar, centered at 10
LIDAR_ANGLE_RANGE = 1.5708  # 90 degrees, 1.5708 radians

# These are your pose values that you will update by solving the odometry equations
pose_x = 2.125
pose_y = 0.125
pose_theta = math.pi

# velocity reduction percent
MAX_VEL_REDUCTION = 0.2  # Run robot at 20% of max speed

# ePuck Constants
EPUCK_AXLE_DIAMETER = 0.053  # ePuck's wheels are 53mm apart.
EPUCK_MAX_WHEEL_SPEED = 0.125 * MAX_VEL_REDUCTION  # To be filled in with ePuck wheel speed in m/s

# Index into ground_sensors and ground_sensor_readings for each of the 3 onboard sensors.
LEFT_IDX = 0
CENTER_IDX = 1
RIGHT_IDX = 2
WHEEL_FORWARD = 1
WHEEL_STOPPED = 0
WHEEL_BACKWARD = -1

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
SIM_TIMESTEP = int(robot.getBasicTimeStep())

# Initialize Motors
leftMotor = robot.getMotor('left wheel motor')
rightMotor = robot.getMotor('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

# get and enable lidar
lidar = robot.getLidar("LDS-01")
lidar.enable(SIM_TIMESTEP)
lidar.enablePointCloud()

# Initialize lidar motors
lidar_main_motor = robot.getMotor('LDS-01_main_motor')
lidar_secondary_motor = robot.getMotor('LDS-01_secondary_motor')
lidar_main_motor.setPosition(float('inf'))
lidar_secondary_motor.setPosition(float('inf'))
lidar_main_motor.setVelocity(30.0)
lidar_secondary_motor.setVelocity(60.0)

# Initialize lidar readings array and offsets
lidar_readings_array = []
start_angle_in_radians = LIDAR_ANGLE_RANGE / 2
angle_offset_in_radians = -LIDAR_ANGLE_RANGE / (LIDAR_ANGLE_BINS - 1)
lidar_offsets_array = [start_angle_in_radians + angle_offset_in_radians * i for i in range(LIDAR_ANGLE_BINS)]

# Map Variables
MAP_BOUNDS = [2.25, 2.25]
CELL_RESOLUTIONS = np.array([0.25, 0.25])  # 10cm per cell
NUM_X_CELLS = int(MAP_BOUNDS[0] / CELL_RESOLUTIONS[0])
NUM_Y_CELLS = int(MAP_BOUNDS[1] / CELL_RESOLUTIONS[1])

world_map = np.zeros([NUM_Y_CELLS, NUM_X_CELLS])


def update_odometry(left_wheel_direction, right_wheel_direction, time_elapsed):
    """
    Given the amount of time passed and the direction each wheel was rotating,
    update the robot's pose information accordingly
    """
    global pose_x, pose_y, pose_theta, EPUCK_MAX_WHEEL_SPEED, EPUCK_AXLE_DIAMETER
    pose_theta += (right_wheel_direction - left_wheel_direction) * time_elapsed * EPUCK_MAX_WHEEL_SPEED / EPUCK_AXLE_DIAMETER
    pose_x += math.cos(pose_theta) * time_elapsed * EPUCK_MAX_WHEEL_SPEED * (left_wheel_direction + right_wheel_direction) / 2.
    pose_y += math.sin(pose_theta) * time_elapsed * EPUCK_MAX_WHEEL_SPEED * (left_wheel_direction + right_wheel_direction) / 2.
    pose_theta = get_bounded_theta(pose_theta)


def get_bounded_theta(theta):
    """
    Returns theta bounded in [-PI, PI]
    """
    while theta > math.pi:
        theta -= 2. * math.pi
    while theta < -math.pi:
        theta += 2. * math.pi
    return theta


def convert_lidar_reading_to_world_coord(lidar_bin, lidar_distance):
    """
    @param lidar_bin: The beam index that provided this measurement
    @param lidar_distance: The distance measurement from the sensor for that beam
    @return world_point: List containing the corresponding (x,y) point in the world frame of reference
    """
    global pose_x, pose_y, pose_theta
    global lidar_offsets_array

    x_robot = lidar_distance * np.cos(lidar_offsets_array[lidar_bin])
    y_robot = lidar_distance * np.sin(lidar_offsets_array[lidar_bin])

    x_world = np.cos(pose_theta) * x_robot - np.sin(pose_theta) * y_robot + pose_x
    y_world = np.sin(pose_theta) * x_robot + np.cos(pose_theta) * y_robot + pose_y

    return x_world, y_world


def transform_world_coord_to_map_coord(world_coord):
    """
    @param world_coord: Tuple of (x,y) position in world coordinates
    @return grid_coord: Tuple of (i,j) coordinates corresponding to grid row (y-coord) and column (x-coord) in our map
    """
    col, row = np.array(world_coord) / CELL_RESOLUTIONS
    if row < 0 or col < 0 or row >= NUM_Y_CELLS or col >= NUM_X_CELLS:
        return None

    return tuple(np.array([row, col]).astype(int))


def transform_map_coord_world_coord(map_coord):
    """
    @param map_coord: Tuple of (i,j) coordinates corresponding to grid column and row in our map
    @return world_coord: Tuple of (x,y) position corresponding to the center of map_coord, in world coordinates
    """
    row, col = map_coord
    if row < 0 or col < 0 or row >= NUM_Y_CELLS or col >= NUM_X_CELLS:
        return None

    return np.array([(col+0.5)*CELL_RESOLUTIONS[1], (row+0.5)*CELL_RESOLUTIONS[0]])


def update_map(lidar_readings_array):
    """
    @param lidar_readings_array
    """
    global world_map

    for k in range(LIDAR_ANGLE_BINS):

        if lidar_readings_array[k] < 0.3:  # LIDAR_SENSOR_MAX_RANGE:

            coords = convert_lidar_reading_to_world_coord(k, lidar_readings_array[k])

            map_coords = transform_world_coord_to_map_coord(coords)

            if map_coords is not None:
                i, j = map_coords

                print(i, j, coords)

                world_map[i, j] = 1


def display_map(m):
    """
    @param m: The world map matrix to visualize
    """
    sns.heatmap(m[::-1, :], cbar=False, xticklabels=False, yticklabels=False)
    plt.show()


last_odometry_update_time = None

# Keep track of which direction each wheel is turning
left_wheel_direction = WHEEL_STOPPED
right_wheel_direction = WHEEL_STOPPED

# Important IK Variable storing final desired pose
target_pose = None  # Populated by the supervisor, only when the target is moved.

# Sensor burn-in period
for i in range(10):
    robot.step(SIM_TIMESTEP)

# Main Control Loop:
while robot.step(SIM_TIMESTEP) != -1:

    # If first time entering the loop, just take the current time as "last odometry update" time
    if last_odometry_update_time is None:
        last_odometry_update_time = robot.getTime()

    # Update Odometry
    time_elapsed = robot.getTime() - last_odometry_update_time
    last_odometry_update_time += time_elapsed
    update_odometry(left_wheel_direction, right_wheel_direction, time_elapsed)

    left_wheel_direction, right_wheel_direction = WHEEL_FORWARD * 0.5, WHEEL_FORWARD * 0.5
    leftMotor.setVelocity(EPUCK_MAX_WHEEL_SPEED * math.pi)
    rightMotor.setVelocity(EPUCK_MAX_WHEEL_SPEED * math.pi)

    # ####
    # # YOUR CODE HERE for Part 1.3, 3.4, and 4.1
    # ####
    # lidar_readings_array = lidar.getRangeImage()
    #
    # # print(lidar_readings_array)
    #
    # pose_x = pose_x
    # pose_y = 1 - pose_y
    # pose_theta = pose_theta + math.pi / 2
    #
    # update_map(lidar_readings_array)

    print((pose_x, pose_y), transform_world_coord_to_map_coord((pose_x, pose_y)))

# Enter here exit cleanup code.
