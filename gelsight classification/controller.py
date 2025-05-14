# #!/usr/bin/env python3
# # import rospy
# import cv2
# import os
# import time
# import random
# import json
# import math
# import setting
# import find_marker
# import A_utility
# from rtde_control import RTDEControlInterface
# from rtde_receive import RTDEReceiveInterface

# # --------------------------
# # Configuration Parameters
# # --------------------------
# ROBOT_IP = "10.10.10.1"
# HOME_WITH_TOOL = [-0.298569, -0.694446, 0.239335, 0.633457, -1.477861, 0.626266]
# NEW_TCP = (0.0, 0.0, 0.26, 0.0, 0.0, 0.0)
# DESCENT_MEAN = 0.01            # Mean descent of 1cm
# DESCENT_RANGE = 0.0015         # ±1.5mm variation
# SLIDE_RANGE = 0.01             # 1cm lateral adjustment range
# TOTAL_CYCLES = 100             # Number of cycles
# RUN_IDENTIFIER = "B"          # Identifier for folder naming
# PARENT_FOLDER = "material_4"   # Parent folder for all data

# # --------------------------
# # Coordinate Transformation
# # --------------------------
# def rotate_xy(x, y, theta):
#     cos_t = math.cos(theta)
#     sin_t = math.sin(theta)
#     return cos_t * x - sin_t * y, sin_t * x + cos_t * y


# def transform_old_to_new(pose_old):
#     x_old, y_old, z, rx, ry, rz = pose_old
#     x_new, y_new = rotate_xy(x_old, y_old, -math.radians(45))
#     return [x_new, y_new, z, rx, ry, rz]


# def transform_new_to_old(pose_new):
#     x_new, y_new, z, rx, ry, rz = pose_new
#     x_old, y_old = rotate_xy(x_new, y_new, math.radians(45))
#     return [x_old, y_old, z, rx, ry, rz]

# # --------------------------
# # ControllerNode: Single-Sensor Collection
# # --------------------------
# class ControllerNode:
#     def __init__(self):
#         # Initialize RTDE interfaces
#         self.rtde_c = RTDEControlInterface(ROBOT_IP)
#         self.rtde_r = RTDEReceiveInterface(ROBOT_IP)
#         self.rtde_c.setTcp(NEW_TCP)
#         rospy.loginfo("TCP set: {}".format(NEW_TCP))
#         time.sleep(0.3)

#         # Move to HOME_WITH_TOOL
#         rospy.loginfo("Moving to HOME_WITH_TOOL...")
#         self.rtde_c.moveL(HOME_WITH_TOOL, 0.2, 0.1)
#         rospy.loginfo("Reached HOME_WITH_TOOL.")

#         # Compute home in new coordinate
#         self.home_new = transform_old_to_new(HOME_WITH_TOOL)

#         # Initialize camera and marker matcher
#         self.cam = cv2.VideoCapture(3)
#         self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
#         self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
#         setting.init()
#         self.marker = find_marker.Matching(
#             N_=setting.N_, M_=setting.M_, fps_=setting.fps_,
#             x0_=setting.x0_, y0_=setting.y0_, dx_=setting.dx_, dy_=setting.dy_
#         )

#     def collect_cycle(self, cycle):
#         # Random lateral adjustment along Y within SLIDE_RANGE
#         random_y = random.uniform(0, SLIDE_RANGE)
#         hover_adj = list(self.home_new)
#         hover_adj[1] += random_y

#         # Move above adjusted position
#         pos_old = transform_new_to_old(hover_adj)
#         self.rtde_c.moveL(pos_old, 0.2, 0.1)
#         rospy.sleep(1)

#         # Random descent amount
#         descent = random.uniform(DESCENT_MEAN - DESCENT_RANGE, DESCENT_MEAN + DESCENT_RANGE)
#         descend_pose = list(hover_adj)
#         descend_pose[2] -= descent

#         # Descend to contact
#         self.rtde_c.moveL(transform_new_to_old(descend_pose), 0.2, 0.1)
#         rospy.sleep(0.5)

#         # Capture image and marker flow
#         frame = A_utility.get_processed_frame(self.cam)
#         centers = A_utility.marker_center(frame)
#         self.marker.init(centers)
#         self.marker.run()
#         flow = self.marker.get_flow()

#         # Save data
#         folder = os.path.join(PARENT_FOLDER, f"cycle_{cycle}_{RUN_IDENTIFIER}")
#         os.makedirs(folder, exist_ok=True)
#         img_path = os.path.join(folder, "frame.png")
#         cv2.imwrite(img_path, frame)
#         with open(os.path.join(folder, "flow.json"), "w") as f:
#             json.dump(flow, f)

#         # Return to hover above position
#         self.rtde_c.moveL(transform_new_to_old(hover_adj), 0.2, 0.1)
#         rospy.sleep(1)

#     def run(self):
#         # Initial descent from home to hover
#         hover = list(self.home_new)
#         hover[2] -= DESCENT_MEAN
#         self.rtde_c.moveL(transform_new_to_old(hover), 0.2, 0.1)
#         rospy.sleep(2)

#         # Ensure parent folder exists
#         os.makedirs(PARENT_FOLDER, exist_ok=True)

#         # Execute cycles
#         for cycle in range(1, TOTAL_CYCLES + 1):
#             rospy.loginfo(f"Starting cycle {cycle}/{TOTAL_CYCLES}")
#             self.collect_cycle(cycle)

#         # Return to HOME and stop
#         rospy.loginfo("All cycles completed. Returning to HOME_WITH_TOOL.")
#         self.rtde_c.moveL(HOME_WITH_TOOL, 0.2, 0.1)
#         rospy.sleep(2)
#         self.rtde_c.servoStop()
#         self.rtde_c.stopScript()


# def main():
#     rospy.init_node("controller_node", anonymous=True)
#     node = ControllerNode()
#     node.run()

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
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
RTDE_PORT       = 50002            # <-- changed RTDE control port
HOME_WITH_TOOL  = [-0.298569, -0.694446, 0.239335, 0.633457, -1.477861, 0.626266]
NEW_TCP         = (0.0, 0.0, 0.26, 0.0, 0.0, 0.0)
DESCENT_MEAN    = 0.018             # Mean descent of 1 cm
DESCENT_RANGE   = 0.0010           # ±1.5 mm variation
SLIDE_RANGE     = 0.05             # 1 cm lateral variation
TOTAL_CYCLES    = 100              # Number of presses
RUN_IDENTIFIER  = "A"              # Identifier for folder naming
PARENT_FOLDER   = "material_10"     # Parent folder for all data

# --------------------------
# Coordinate Transformation
# --------------------------
def rotate_xy(x, y, theta):
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return cos_t * x - sin_t * y, sin_t * x + cos_t * y

def transform_old_to_new(pose_old):
    x_old, y_old, z, rx, ry, rz = pose_old
    x_new, y_new = rotate_xy(x_old, y_old, -math.radians(45))
    return [x_new, y_new, z, rx, ry, rz]

def transform_new_to_old(pose_new):
    x_new, y_new, z, rx, ry, rz = pose_new
    x_old, y_old = rotate_xy(x_new, y_new, math.radians(45))
    return [x_old, y_old, z, rx, ry, rz]

# --------------------------
# Controller: Single-Sensor Collection
# --------------------------
class ControllerNode:
    def __init__(self):
        # Initialize RTDE interfaces (control on port 50002)
        self.rtde_c = RTDEControlInterface(ROBOT_IP, RTDE_PORT)
        self.rtde_r = RTDEReceiveInterface(ROBOT_IP)
        
        # Configure TCP
        self.rtde_c.setTcp(NEW_TCP)
        print(f"[INFO] TCP set to: {NEW_TCP}")
        time.sleep(0.3)

        # Move to HOME_WITH_TOOL
        print("[INFO] Moving to HOME_WITH_TOOL...")
        self.rtde_c.moveL(HOME_WITH_TOOL, 0.2, 0.1)
        print("[INFO] Reached HOME_WITH_TOOL.")

        # Compute home in new coordinate system
        self.home_new = transform_old_to_new(HOME_WITH_TOOL)

        # Initialize camera and marker matcher
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        setting.init()
        self.marker = find_marker.Matching(
            N_=setting.N_, M_=setting.M_, fps_=setting.fps_,
            x0_=setting.x0_, y0_=setting.y0_, dx_=setting.dx_, dy_=setting.dy_
        )

    def collect_cycle(self, cycle):
        # Random lateral adjustment along Y within SLIDE_RANGE
        random_y = random.uniform(0, SLIDE_RANGE)
        hover_adj = list(self.home_new)
        hover_adj[1] += random_y

        # Move above adjusted position
        pos_old = transform_new_to_old(hover_adj)
        self.rtde_c.moveL(pos_old, 0.2, 0.1)
        time.sleep(1)

        # Random descent amount
        descent = random.uniform(DESCENT_MEAN - DESCENT_RANGE, DESCENT_MEAN + DESCENT_RANGE)
        descend_pose = list(hover_adj)
        descend_pose[2] -= descent

        # Descend to contact
        self.rtde_c.moveL(transform_new_to_old(descend_pose), 0.2, 0.1)
        time.sleep(0.5)

        # Capture image and marker flow
        frame = A_utility.get_processed_frame(self.cam)
        centers = A_utility.marker_center(frame)
        self.marker.init(centers)
        self.marker.run()
        flow = self.marker.get_flow()

        # Save data
        folder = os.path.join(PARENT_FOLDER, f"cycle_{cycle}_{RUN_IDENTIFIER}")
        os.makedirs(folder, exist_ok=True)
        cv2.imwrite(os.path.join(folder, "frame.png"), frame)
        with open(os.path.join(folder, "flow.json"), "w") as f:
            json.dump(flow, f)

        # Return to hover above position
        self.rtde_c.moveL(transform_new_to_old(hover_adj), 0.2, 0.1)
        time.sleep(1)

    def run(self):
        # Initial descent from home to hover
        hover = list(self.home_new)
        hover[2] -= DESCENT_MEAN
        self.rtde_c.moveL(transform_new_to_old(hover), 0.2, 0.1)
        time.sleep(2)

        # Ensure parent folder exists
        os.makedirs(PARENT_FOLDER, exist_ok=True)

        # Execute cycles
        for cycle in range(1, TOTAL_CYCLES + 1):
            print(f"[INFO] Starting cycle {cycle}/{TOTAL_CYCLES}")
            self.collect_cycle(cycle)

        # Return to HOME and stop motion
        print("[INFO] All cycles completed. Returning to HOME_WITH_TOOL.")
        self.rtde_c.moveL(HOME_WITH_TOOL, 0.2, 0.1)
        time.sleep(2)
        self.rtde_c.servoStop()
        self.rtde_c.stopScript()

def main():
    node = ControllerNode()
    node.run()

if __name__ == "__main__":
    main()
