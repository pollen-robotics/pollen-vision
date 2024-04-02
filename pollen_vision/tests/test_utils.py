from pathlib import Path

import depthai as dai
import numpy as np
from pollen_vision.camera_wrappers.depthai.cam_config import CamConfig
from pollen_vision.camera_wrappers.depthai.utils import (
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
