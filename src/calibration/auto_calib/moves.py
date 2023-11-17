import os
import pickle
import time

import cv2
from cv2 import aruco
from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode

from depthai_wrappers.cv_wrapper import CvWrapper
from depthai_wrappers.utils import drawEpiLines


def get_board(reachy: ReachySDK) -> bool:
    OK = False
    while not OK:
        goto(
            goal_positions={reachy.r_arm.r_gripper: -50.0},
            duration=1.0,
            interpolation_mode=InterpolationMode.MINIMUM_JERK,
        )
        right_angled_position: dict = {
            reachy.r_arm.r_shoulder_pitch: 0,
            reachy.r_arm.r_shoulder_roll: 0,
            reachy.r_arm.r_arm_yaw: 0,
            reachy.r_arm.r_elbow_pitch: -90,
            reachy.r_arm.r_forearm_yaw: 0,
            reachy.r_arm.r_wrist_pitch: 0,
            reachy.r_arm.r_wrist_roll: 0,
        }
        goto(
            goal_positions=right_angled_position,
            duration=2.0,
            interpolation_mode=InterpolationMode.MINIMUM_JERK,
        )
        print("place the board in the gripper")
        input("Press any key to close the gripper")
        goto(
            goal_positions={reachy.r_arm.r_gripper: 20.0},
            duration=1.0,
            interpolation_mode=InterpolationMode.MINIMUM_JERK,
        )
        res = input("Is the board in the gripper? (y/n)")
        while res.lower() not in ["y", "n"]:
            res = input("Is the board in the gripper? (y/n)")

        if res == "y":
            print("Board acquired, proceeding")
            OK = True
        else:
            res = input("Board not acquired, try again ? (y/n)")
            while res.lower() not in ["y", "n"]:
                res = input("Board not acquired, try again ? (y/n)")
            if res == "n":
                print("Cancelling ...")
                return False

    return True


def replay_recording(
    reachy: ReachySDK,
    recording_path: str,
    w: CvWrapper,
    left_path: str,
    right_path: str,
    acquire: bool = False,
    check_epilines: bool = False,
    duration: int = 3.0,
) -> None:
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
    recording = pickle.load(open(recording_path, "rb"))

    for i, point in enumerate(recording):
        print(str(i), "out of ", str(len(recording)))
        p = dict(zip(recorded_joints, point))
        goto(p, duration=duration)
        if acquire:
            time.sleep(1)
            data, _, _ = w.get_data()
            cv2.imwrite(os.path.join(left_path, str(i) + ".png"), data["left"])
            cv2.imwrite(os.path.join(right_path, str(i) + ".png"), data["right"])
        elif check_epilines:
            data, _, _ = w.get_data()
            concatim, score = drawEpiLines(
                data["left"],
                data["right"],
                aruco.getPredefinedDictionary(aruco.DICT_4X4_1000),
            )
            print("Score: ", score)
