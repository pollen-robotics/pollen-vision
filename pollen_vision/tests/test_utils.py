from pathlib import Path

import cv2
import depthai as dai
import numpy as np
from cv2 import aruco
from pollen_vision.camera_wrappers.depthai.cam_config import CamConfig
from pollen_vision.camera_wrappers.depthai.utils import (
    drawEpiLines,
    get_config_file_path,
    get_inv_R_T,
    get_socket_from_name,
)


def test_inv() -> None:
    R = np.array([[0.36, 0.48, -0.8], [-0.8, 0.6, 0], [0.46, 0.64, 0.6]])
    T = np.array([10, 11, 12])

    invR, invT = get_inv_R_T(R, T)
    R_ref = np.array(
        [[0.3634895, -0.80775444, 0.48465267], [0.48465267, 0.58966074, 0.64620355], [-0.79563813, -0.00969305, 0.60581583]]
    )

    T_ref = np.array([-0.56542811, -19.08723748, 0.79321486])

    assert np.allclose(invR, R_ref)
    assert np.allclose(invT, T_ref)


def test_socket_names() -> None:
    path = str((Path(__file__).parent / "data" / Path("calibration_unit_tests.json")).resolve().absolute())
    c = CamConfig(get_config_file_path("CONFIG_OAK_D_PRO"), 60, resize=(1280, 720), exposure_params=None)
    c.calib = dai.CalibrationHandler(path)

    left_socket = get_socket_from_name("left", c.name_to_socket)
    right_socket = get_socket_from_name("right", c.name_to_socket)

    assert left_socket == dai.CameraBoardSocket.CAM_B
    assert right_socket == dai.CameraBoardSocket.CAM_C


def test_draw_epilines() -> None:
    path_left = str((Path(__file__).parent / "data" / Path("left_epiline.jpg")).resolve().absolute())
    path_right = str((Path(__file__).parent / "data" / Path("right_epiline.jpg")).resolve().absolute())
    path_epilines = str((Path(__file__).parent / "data" / Path("epilines.npz")).resolve().absolute())

    epilines_ref = np.load(path_epilines)

    imleft = cv2.imread(path_left)
    imright = cv2.imread(path_right)
    ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)

    epilines = drawEpiLines(imleft, imright, ARUCO_DICT)
    assert np.array_equal(epilines, epilines_ref["epilines"])
