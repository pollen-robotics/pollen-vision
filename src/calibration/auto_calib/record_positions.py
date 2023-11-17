import argparse
import pickle

import cv2
import numpy as np
from reachy_sdk import ReachySDK

from depthai_wrappers.cv_wrapper import CvWrapper

argParser = argparse.ArgumentParser(
    description="Acquire images from a luxonis camera and save them to disk."
)
argParser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to the configuration file.",
)
argParser.add_argument(
    "--recordsPath",
    type=str,
    default="./recordings.pkl",
)
args = argParser.parse_args()

positions = []
reachy = ReachySDK(host="192.168.1.252")

recorded_joints = [
    reachy.r_arm.r_shoulder_pitch,
    reachy.r_arm.r_shoulder_roll,
    reachy.r_arm.r_arm_yaw,
    reachy.r_arm.r_elbow_pitch,
    reachy.r_arm.r_forearm_yaw,
    reachy.r_arm.r_wrist_pitch,
    reachy.r_arm.r_wrist_roll,
    reachy.head.neck_roll,
    reachy.head.neck_pitch,
    reachy.head.neck_yaw,
    reachy.head.l_antenna,
    reachy.head.r_antenna,
]

reachy.r_arm.r_gripper.compliant = False
reachy.r_arm.r_gripper.goal_position = -50
input("Place the board in the gripper and press any key")
reachy.r_arm.r_gripper.goal_position = 20
w = CvWrapper(args.config, rectify=False, fps=20)
print(
    "set the right arm and head at the desired position then enter. Or press q to quit"
)
while True:
    data, _, _ = w.get_data()
    l_im = data["left"]
    r_im = data["right"]
    concatim = cv2.resize(np.hstack((l_im, r_im)), (0, 0), fx=0.7, fy=0.7)
    cv2.imshow("concat", concatim)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == 13:
        current_point = [joint.present_position for joint in recorded_joints]
        print(current_point)
        positions.append(current_point)
        pickle.dump(positions, open(args.recordsPath, "wb"))

print("DONE")
