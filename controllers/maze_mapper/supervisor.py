"""supervisor controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
import copy
from controller import Supervisor
import numpy as np
import math

supervisor = None
robot_node = None


def init_supervisor():
    global supervisor, robot_node

    # create the Supervisor instance.
    supervisor = Supervisor()

    # do this once only
    root = supervisor.getRoot()
    root_children_field = root.getField("children")
    robot_node = None
    for idx in range(root_children_field.getCount()):
        if root_children_field.getMFNode(idx).getDef() == "EPUCK":
            robot_node = root_children_field.getMFNode(idx)

    start_translation = copy.copy(robot_node.getField("translation").getSFVec3f())
    start_rotation = copy.copy(robot_node.getField("rotation").getSFRotation())


def supervisor_get_robot_pose():
    """
    Returns robot position
    """
    robot_position = np.array(robot_node.getField("translation").getSFVec3f())

    robot_pose = np.array(
        [robot_position[0], robot_position[2], robot_node.getField("rotation").getSFRotation()[3]])
    return robot_pose
