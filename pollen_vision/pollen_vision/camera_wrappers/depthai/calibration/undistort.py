from typing import List, Tuple

import cv2
import depthai as dai
import numpy as np
import numpy.typing as npt
from pollen_vision.camera_wrappers.depthai.cam_config import CamConfig
from pollen_vision.camera_wrappers.depthai.utils import get_socket_from_name


def compute_undistort_maps(
    cam_config: CamConfig,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
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
        mapXL, mapYL = cv2.fisheye.initUndistortRectifyMap(left_K, left_D, R1, P1, resolution, cv2.CV_32FC1)
        mapXR, mapYR = cv2.fisheye.initUndistortRectifyMap(right_K, right_D, R2, P2, resolution, cv2.CV_32FC1)
    else:
        mapXL, mapYL = cv2.initUndistortRectifyMap(left_K, left_D, R1, P1, resolution, cv2.CV_32FC1)
        mapXR, mapYR = cv2.initUndistortRectifyMap(right_K, right_D, R2, P2, resolution, cv2.CV_32FC1)

    return mapXL.astype(np.float32), mapYL.astype(np.float32), mapXR.astype(np.float32), mapYR.astype(np.float32)


def get_mesh(cam_config: CamConfig, cam_name: str) -> Tuple[List[dai.Point2f], int, int]:
    """Computes and returns the mesh for the rectification.
    This mesh is used by setWarpMesh in the imageManip nodes.
    """

    if cam_config.undistort_maps[cam_name] is None:
        raise Exception("Undistort maps have not been computed. Call compute_undistort_maps() first.")

    mapX, mapY = cam_config.undistort_maps[cam_name]

    meshCellSize = 16
    mesh0 = []
    for y in range(mapX.shape[0] + 1):
        if y % meshCellSize == 0:
            rowLeft = []
            for x in range(mapX.shape[1]):
                if x % meshCellSize == 0:
                    if y == mapX.shape[0] and x == mapX.shape[1]:
                        rowLeft.append(mapX[y - 1, x - 1])
                        rowLeft.append(mapY[y - 1, x - 1])
                    elif y == mapX.shape[0]:
                        rowLeft.append(mapX[y - 1, x])
                        rowLeft.append(mapY[y - 1, x])
                    elif x == mapX.shape[1]:
                        rowLeft.append(mapX[y, x - 1])
                        rowLeft.append(mapY[y, x - 1])
                    else:
                        rowLeft.append(mapX[y, x])
                        rowLeft.append(mapY[y, x])
            if (mapX.shape[1] % meshCellSize) % 2 != 0:
                rowLeft.append(0)
                rowLeft.append(0)

            mesh0.append(rowLeft)

    mesh0_tmp = np.array(mesh0)
    meshWidth = mesh0_tmp.shape[1] // 2
    meshHeight = mesh0_tmp.shape[0]
    mesh0_tmp.resize(meshWidth * meshHeight, 2)

    mesh = list(map(tuple, mesh0_tmp))

    return mesh, meshWidth, meshHeight  # type: ignore[return-value]
