#!/usr/bin/env python3
import rclpy
# from rclpy.node import Node
import cv2
import os
import time
import random
import json
import math

import setting
import find_marker
import A_utility
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

# --------------------------
# Configuration Parameters
# --------------------------
ROBOT_IP        = "10.10.10.1"
HOME_WITH_TOOL  = [-0.298569, -0.694446, 0.239335, 0.633457, -1.477861, 0.626266]
NEW_TCP         = (0.0, 0.0, 0.26, 0.0, 0.0, 0.0)
DESCENT_MEAN    = 0.01      # Mean descent of 1 cm
DESCENT_RANGE   = 0.0015    # ±1.5 mm variation
SLIDE_RANGE     = 0.01      # 1 cm lateral variation
TOTAL_CYCLES    = 100       # Number of presses
RUN_IDENTIFIER  = "B"       # For folder naming
PARENT_FOLDER   = "material_4"
CUSTOM_PORT = 50002


# --------------------------
# Coordinate transforms
# --------------------------
def rotate_xy(x: float, y: float, theta: float):
    c, s = math.cos(theta), math.sin(theta)
    return c*x - s*y, s*x + c*y

def transform_old_to_new(pose_old):
    x, y, z, rx, ry, rz = pose_old
    xn, yn = rotate_xy(x, y, -math.radians(45))
    return [xn, yn, z, rx, ry, rz]

def transform_new_to_old(pose_new):
    x, y, z, rx, ry, rz = pose_new
    xo, yo = rotate_xy(x, y, math.radians(45))
    return [xo, yo, z, rx, ry, rz]


# --------------------------
# ControllerNode for ROS 2
# --------------------------
class ControllerNode(Node):
    def __init__(self):
        super().__init__("controller_node")
        # RTDE interfaces
        self.rtde_c = RTDEControlInterface(
            ROBOT_IP,
            ur_cap_port=CUSTOM_PORT
        )
        # RTDE receive on custom port (port kwarg is correct here)
        self.rtde_r = RTDEReceiveInterface(
            ROBOT_IP,
            port=CUSTOM_PORT
        )

        # Configure tool center point
        self.rtde_c.setTcp(NEW_TCP)
        self.get_logger().info(f"TCP set: {NEW_TCP}")
        time.sleep(0.3)

        # Move to home
        self.get_logger().info("Moving to HOME_WITH_TOOL…")
        self.rtde_c.moveL(HOME_WITH_TOOL, 0.2, 0.1)
        self.get_logger().info("Reached HOME_WITH_TOOL.")

        # Precompute home in rotated coords
        self.home_new = transform_old_to_new(HOME_WITH_TOOL)

        # Camera + marker matcher
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

        setting.init()
        self.matcher = find_marker.Matching(
            N_=setting.N_,
            M_=setting.M_,
            fps_=setting.fps_,
            x0_=setting.x0_,
            y0_=setting.y0_,
            dx_=setting.dx_,
            dy_=setting.dy_,
        )

    def collect_cycle(self, cycle: int):
        self.get_logger().info(f"Starting cycle {cycle}/{TOTAL_CYCLES}")
        # random lateral shift
        rand_y = random.uniform(0, SLIDE_RANGE)
        hover = list(self.home_new)
        hover[1] += rand_y

        # move above
        self.rtde_c.moveL(transform_new_to_old(hover), 0.2, 0.1)
        time.sleep(1.0)

        # descend randomly
        descent = random.uniform(DESCENT_MEAN - DESCENT_RANGE,
                                 DESCENT_MEAN + DESCENT_RANGE)
        descend = list(hover)
        descend[2] -= descent
        self.rtde_c.moveL(transform_new_to_old(descend), 0.2, 0.1)
        time.sleep(0.5)

        # capture and flow
        frame = A_utility.get_processed_frame(self.cam)
        centers = A_utility.marker_center(frame)
        self.matcher.init(centers)
        self.matcher.run()
        flow = self.matcher.get_flow()

        # save
        outdir = os.path.join(PARENT_FOLDER, f"cycle_{cycle}_{RUN_IDENTIFIER}")
        os.makedirs(outdir, exist_ok=True)
        cv2.imwrite(os.path.join(outdir, "frame.png"), frame)
        with open(os.path.join(outdir, "flow.json"), "w") as f:
            json.dump(flow, f)

        # retreat to hover
        self.rtde_c.moveL(transform_new_to_old(hover), 0.2, 0.1)
        time.sleep(1.0)

    def run(self):
        # initial hover
        hover = list(self.home_new)
        hover[2] -= DESCENT_MEAN
        self.rtde_c.moveL(transform_new_to_old(hover), 0.2, 0.1)
        time.sleep(2.0)

        os.makedirs(PARENT_FOLDER, exist_ok=True)

        for c in range(1, TOTAL_CYCLES + 1):
            self.collect_cycle(c)

        # go home and finish
        self.get_logger().info("All cycles done. Returning home.")
        self.rtde_c.moveL(HOME_WITH_TOOL, 0.2, 0.1)
        time.sleep(2.0)
        self.rtde_c.servoStop()
        self.rtde_c.stopScript()


def main():
    rclpy.init()
    node = ControllerNode()
    try:
        node.run()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
