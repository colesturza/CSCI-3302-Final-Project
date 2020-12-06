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
pose_x = 0.25
pose_y = 0.25
pose_theta = 0

# velocity reduction percent
MAX_VEL_REDUCTION = 1 # 0.2  # Run robot at 20% of max speed

# ePuck Constants
EPUCK_AXLE_DIAMETER = 0.053  # ePuck's wheels are 53mm apart.
EPUCK_MAX_WHEEL_SPEED = .125 * MAX_VEL_REDUCTION  # To be filled in with ePuck wheel speed in m/s

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

lidar_readings_array = []
start_angle_in_radians = LIDAR_ANGLE_RANGE/2
angle_offset_in_radians = -LIDAR_ANGLE_RANGE/(LIDAR_ANGLE_BINS-1)
lidar_offsets_array = [start_angle_in_radians + angle_offset_in_radians * i for i in range(LIDAR_ANGLE_BINS)]


def update_odometry(left_wheel_direction, right_wheel_direction, time_elapsed):
    """
    Given the amount of time passed and the direction each wheel was rotating,
    update the robot's pose information accordingly
    """
    global pose_x, pose_y, pose_theta, EPUCK_MAX_WHEEL_SPEED, EPUCK_AXLE_DIAMETER
    pose_theta += (right_wheel_direction - left_wheel_direction) * time_elapsed * EPUCK_MAX_WHEEL_SPEED / EPUCK_AXLE_DIAMETER;
    pose_x += math.cos(pose_theta) * time_elapsed * EPUCK_MAX_WHEEL_SPEED * (left_wheel_direction + right_wheel_direction)/2.;
    pose_y += math.sin(pose_theta) * time_elapsed * EPUCK_MAX_WHEEL_SPEED * (left_wheel_direction + right_wheel_direction)/2.;
    pose_theta = get_bounded_theta(pose_theta)


def get_bounded_theta(theta):
    """
    Returns theta bounded in [-PI, PI]
    """
    while theta > math.pi: theta -= 2.*math.pi
    while theta < -math.pi: theta += 2.*math.pi
    return theta


########
# Part 1.1 - 1.2: Initialize your LIDAR-related data structures here
########
lidar_readings_array = []
start_angle_in_radians = LIDAR_ANGLE_RANGE/2
angle_offset_in_radians = -LIDAR_ANGLE_RANGE/(LIDAR_ANGLE_BINS-1)
lidar_offsets_array = [start_angle_in_radians + angle_offset_in_radians * i for i in range(LIDAR_ANGLE_BINS)]

########
# Part 3.1: Initialize your map data structure here
########
NUMBER_OF_CELLS = 9
WORLD_MAP_RESOLUTION = 4.5/NUMBER_OF_CELLS
world_map = np.zeros((NUMBER_OF_CELLS, NUMBER_OF_CELLS))

###
# Part 2.2
###
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


###
# Part 3.2
###
def transform_world_coord_to_map_coord(world_coord):
    """
    @param world_coord: Tuple of (x,y) position in world coordinates
    @return grid_coord: Tuple of (i,j) coordinates corresponding to grid column and row in our map
    """
    global NUMBER_OF_CELLS, WORLD_MAP_RESOLUTION
    x, y = world_coord

    if (x < 0 or x >= 2.25) or (y > 0 or y <= -2.25):
        return None

    j = int(x / WORLD_MAP_RESOLUTION)
    # j -= 1 if j == NUMBER_OF_CELLS else 0  # if x == 1 need to subtract the map coord by 1

    i = -int(y / WORLD_MAP_RESOLUTION)
    # i -= 1 if i == NUMBER_OF_CELLS else 0  # if x == 1 need to subtract the map coord by 1

    return i, j


###
# Part 3.3
###
def transform_map_coord_world_coord(map_coord):
    """
    @param map_coord: Tuple of (i,j) coordinates corresponding to grid column and row in our map
    @return world_coord: Tuple of (x,y) position corresponding to the center of map_coord, in world coordinates
    """
    global NUMBER_OF_CELLS, WORLD_MAP_RESOLUTION
    i, j = map_coord

    if (i < 0 or i >= NUMBER_OF_CELLS) or (j < 0 or j >= NUMBER_OF_CELLS):
        return None

    x = j * WORLD_MAP_RESOLUTION + WORLD_MAP_RESOLUTION/2

    y = i * WORLD_MAP_RESOLUTION + WORLD_MAP_RESOLUTION / 2

    return x, y


###
# Part 3.5
###
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

                world_map[i, j] = 1


###
# Part 4.1
###
def display_map(m):
    """
    @param m: The world map matrix to visualize
    """
    sns.heatmap(m[::-1, :], cbar=False, xticklabels=False, yticklabels=False)
    plt.show()


################################################################################
# Do not modify:
################################################################################
# Odometry variables
last_odometry_update_time = None
loop_closure_detection_time = 0

# Variables to keep track of which direction each wheel is turning for odometry
left_wheel_direction = 0
right_wheel_direction = 0

# Burn a couple cycles waiting to let everything come online
for i in range(10):
    robot.step(SIM_TIMESTEP)

number_of_loop_closers = 0

################################################################################
# Do not modify ^
################################################################################

# Main Control Loop:
while robot.step(SIM_TIMESTEP) != -1:

    # If first time entering the loop, just take the current time as "last odometry update" time
    if last_odometry_update_time is None:
        last_odometry_update_time = robot.getTime()

    # Update Odometry
    time_elapsed = robot.getTime() - last_odometry_update_time
    last_odometry_update_time += time_elapsed
    update_odometry(left_wheel_direction, right_wheel_direction, time_elapsed)

    left_wheel_direction, right_wheel_direction = 1, 1
    leftMotor.setVelocity(EPUCK_MAX_WHEEL_SPEED)
    rightMotor.setVelocity(EPUCK_MAX_WHEEL_SPEED)

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

    print(transform_world_coord_to_map_coord((pose_x, pose_y)))


# Enter here exit cleanup code.
