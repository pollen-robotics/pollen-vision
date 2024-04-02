from typing import Tuple

import cv2
import numpy as np
from pollen_vision.camera_wrappers.depthai.cam_config import CamConfig
from pollen_vision.camera_wrappers.depthai.utils import get_socket_from_name


def compute_undistort_maps(cam_config: CamConfig) -> Tuple[cv2.UMat, cv2.UMat, cv2.UMat, cv2.UMat]:
    """Pre-computes the undistort maps for the rectification."""

    left_socket = get_socket_from_name("left", cam_config.name_to_socket)
    right_socket = get_socket_from_name("right", cam_config.name_to_socket)

    resolution = cam_config.undistort_resolution

    calib = cam_config.get_calib()

    left_K = np.array(
        calib.getCameraIntrinsics(
            left_socket,
            resolution[0],
            resolution[1],
        )
    )
    left_D = np.array(calib.getDistortionCoefficients(left_socket))

    right_K = np.array(
        calib.getCameraIntrinsics(
            right_socket,
            resolution[0],
            resolution[1],
        )
    )
    right_D = np.array(calib.getDistortionCoefficients(right_socket))
    R = np.array(calib.getStereoRightRectificationRotation())

    T = np.array(calib.getCameraTranslationVector(left_socket, right_socket))
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        left_K,
        left_D,
        right_K,
        right_D,
        resolution,
        R,
        T,
        flags=0,
    )

    if cam_config.fisheye:
        # 5 is the value of cv2.CV_32FC1. mypy does not know about this value
        mapXL, mapYL = cv2.fisheye.initUndistortRectifyMap(left_K, left_D, R1, P1, resolution, 5)
        mapXR, mapYR = cv2.fisheye.initUndistortRectifyMap(right_K, right_D, R2, P2, resolution, 5)
    else:
        mapXL, mapYL = cv2.initUndistortRectifyMap(left_K, left_D, R1, P1, resolution, 5)
        mapXR, mapYR = cv2.initUndistortRectifyMap(right_K, right_D, R2, P2, resolution, 5)

    # self.cam_config.set_undistort_maps(mapXL, mapYL, mapXR, mapYR)

    print("here", mapXL)

    return mapXL, mapYL, mapXR, mapYR
