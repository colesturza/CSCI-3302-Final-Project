"""robot_arm_test controller."""

from controller import Robot, DistanceSensor, RangeFinder

pose_x = 0
pose_y = 0
pose_theta = 0

# create the Robot instance
robot = Robot()

# get the time step of the current world
SIM_TIMESTEP = int(robot.getBasicTimeStep())

LIDAR_SENSOR_MAX_RANGE = 3.  # meters
LIDAR_ANGLE_BINS = 51  # 51 bins to cover the angular range of the lidar, centered at 10
LIDAR_ANGLE_RANGE = 1.5708  # 90 degrees, 1.5708 radians

# get and enable lidar 
lidar = robot.getLidar("LDS-01")
lidar.enable(SIM_TIMESTEP)
lidar.enablePointCloud()

# initialize lidar motors
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

# enable the kinect camera and range finder
kinect_color = robot.getCamera('kinect color')
kinect_range = robot.getRangeFinder('kinect range')
kinect_color.enable(SIM_TIMESTEP)
kinect_range.enable(SIM_TIMESTEP)

print([(device.getName(), device.getModel()) for device in [robot.getDeviceByIndex(i) for i in range(robot.getNumberOfDevices())]])

wheels = []
# wheel1 = front right, wheel2 = front left, wheel3 = back right, wheel4 = back left
wheelsNames = ['wheel1', 'wheel2', 'wheel3', 'wheel4']
for i in range(4):
    wheels.append(robot.getMotor(wheelsNames[i]))
    wheels[i].setPosition(float('inf'))
    wheels[i].setVelocity(0.0)

print(wheels[0].getMaxVelocity())


def move_forward():
    global wheels
    wheels[0].setVelocity(0.5)
    wheels[1].setVelocity(0.5)
    wheels[2].setVelocity(0.5)
    wheels[3].setVelocity(0.5)


def move_right():
    global wheels
    wheels[0].setVelocity(-0.5)
    wheels[1].setVelocity(0.5)
    wheels[2].setVelocity(0.5)
    wheels[3].setVelocity(-0.5)


def rotate_clockwise():
    global wheels
    wheels[0].setVelocity(-0.5)
    wheels[1].setVelocity(0.5)
    wheels[2].setVelocity(-0.5)
    wheels[3].setVelocity(0.5)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(SIM_TIMESTEP) != -1:

    lidar_readings_array = lidar.getRangeImage()

    rotate_clockwise()

# Enter here exit cleanup code.
