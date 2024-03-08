# TODO Outdated

import argparse
import time
from typing import Tuple

import cv2
import FramesViewer.utils as fv_utils
import numpy as np
import numpy.typing as npt
from FramesViewer.viewer import Viewer
from pollen_vision.camera_wrappers.depthai import SDKWrapper
from pollen_vision.camera_wrappers.depthai.utils import (
    get_config_file_path,
    get_config_files_names,
)
from pollen_vision.vision_models.object_detection import OwlVitWrapper
from pollen_vision.vision_models.object_segmentation import MobileSamWrapper
from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto

valid_configs = get_config_files_names()
argParser = argparse.ArgumentParser(description="Basic grasping demo")
argParser.add_argument(
    "--config",
    type=str,
    required=True,
    choices=valid_configs,
    help=f"Configutation file name : {valid_configs}",
)
args = argParser.parse_args()

w = SDKWrapper(get_config_file_path(args.config), compute_depth=True)
MSW = MobileSamWrapper()
OW = OwlVitWrapper()

fv = Viewer()
fv.start()
K = w.cam_config.get_K_left()

reachy = ReachySDK("10.0.0.93")
# reachy = ReachySDK("localhost")
reachy.turn_on("r_arm")


def open_gripper() -> None:
    goto({reachy.r_arm.r_gripper: -50}, duration=1.0)


def close_gripper() -> None:
    goto({reachy.r_arm.r_gripper: 0}, duration=1.0)


start_pose = fv_utils.make_pose([0.17, -0.16, -0.3], [0, -90, 0])
joint_start_pose = reachy.r_arm.inverse_kinematics(start_pose)


def goto_start_pose() -> None:
    goto({joint: pos for joint, pos in zip(reachy.r_arm.joints.values(), joint_start_pose)}, duration=2.0)
    open_gripper()


goto_start_pose()


def uv_to_xyz(z: float, u: float, v: float, K: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    cx = K[0, 2]
    cy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    # Calcul des coordonnées dans le monde
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.array([x, y, z])


def get_centroid(mask: npt.NDArray[np.uint8]) -> Tuple[int, int]:
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise Exception("No contours found")
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        raise Exception("No contours found")
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY


T_world_cam = fv_utils.make_pose([0.03, -0.15, 0.1], [0, 0, 0])
# T_world_cam = fv_utils.make_pose([0.0537, -0.1281, 0.1413], [0, 0, 0]) # Old mounting piece
T_world_cam[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
T_world_cam = fv_utils.rotateInSelf(T_world_cam, [-45, 0, 0])
fv.pushFrame(T_world_cam, "camera")

fv.createMesh(
    "gripper",
    "example_meshes/gripper_simplified.obj",
    fv_utils.make_pose([0, 0, 0], [0, 0, 0]),
    scale=10.0,
    wireFrame=True,
)


def grasp(T_world_mug: npt.NDArray[np.float32]) -> None:
    # Proceed to grasp
    pregrasp_pose = T_world_mug.copy()
    pregrasp_pose[:3, :3] = start_pose[:3, :3]
    pregrasp_pose[:3, 3] += np.array([-0.1, 0, 0.05])
    fv.pushFrame(pregrasp_pose, "pregrasp", color=(0, 255, 0))

    joint_pregrasp_pose = reachy.r_arm.inverse_kinematics(pregrasp_pose)
    goto({joint: pos for joint, pos in zip(reachy.r_arm.joints.values(), joint_pregrasp_pose)}, duration=1.0)

    grasp_pose = pregrasp_pose.copy()
    grasp_pose[:3, 3] += np.array([0.15, 0, 0])

    joint_grasp_pose = reachy.r_arm.inverse_kinematics(grasp_pose)
    goto({joint: pos for joint, pos in zip(reachy.r_arm.joints.values(), joint_grasp_pose)}, duration=1.0)
    close_gripper()

    lift_pose = grasp_pose.copy()
    lift_pose[:3, 3] += np.array([0, 0, 0.1])
    joint_lift_pose = reachy.r_arm.inverse_kinematics(lift_pose)
    goto({joint: pos for joint, pos in zip(reachy.r_arm.joints.values(), joint_lift_pose)}, duration=1.0)

    goto({joint: pos for joint, pos in zip(reachy.r_arm.joints.values(), joint_grasp_pose)}, duration=1.0)

    open_gripper()

    goto_start_pose()


try:
    while True:
        print("")
        print("")
        print("")
        object_to_grasp = input("what do you want to grasp ? : ")
        print("press esc to cancel")
        while True:
            T_world_hand = reachy.r_arm.forward_kinematics()
            fv.pushFrame(T_world_hand, "hand")
            fv.updateMesh("gripper", T_world_hand)

            data, _, _ = w.get_data()
            im = data["left"]
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            depth = data["depth"]

            predictions = OW.infer(im, object_to_grasp)
            if len(predictions) != 0:
                bboxes = OW.get_bboxes(predictions)
                labels = OW.get_labels(predictions)
                mask = MSW.infer(im, bboxes)[0]
                depth[mask == 0] = 0
                average_depth = depth[depth != 0].mean()
                try:
                    u, v = get_centroid(mask)
                except Exception as e:
                    print(e)
                    break
                xyz = uv_to_xyz(average_depth * 0.1, u, v, K)
                xyz *= 0.01

                T_cam_mug = fv_utils.make_pose(xyz, [0, 0, 0])
                T_world_mug = T_world_cam @ T_cam_mug
                T_world_mug[:3, :3] = np.eye(3)
                fv.pushFrame(T_world_mug, "mug")

                print(object_to_grasp, "detected !")

                im = MSW.annotate(im, [mask], bboxes, labels, labels_colors=OW.labels_colors)
                im = cv2.circle(im, (u, v), 5, (0, 255, 0), -1)
                cv2.imshow("masked_depth", depth * 255)
                cv2.imshow("im", np.array(cv2.cvtColor(im, cv2.COLOR_RGB2BGR)))
                print("press enter to grasp, any other key to cancel")
                key = cv2.waitKey(0)
                if key == 13:
                    grasp(T_world_mug)
                    break
                else:
                    break

            cv2.imshow("im", np.array(cv2.cvtColor(im, cv2.COLOR_RGB2BGR)))
            key = cv2.waitKey(1)
            if key == 27:
                break

except KeyboardInterrupt:
    print("")
    print("")
    print("")
    print("TURNING OFF")
    reachy.turn_off_smoothly("r_arm")
    time.sleep(3)
    print("DONE")
    exit()
