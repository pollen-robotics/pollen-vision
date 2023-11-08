from typing import Tuple

import depthai as dai
import numpy as np

socket_stringToCam = {
    "CAM_A": dai.CameraBoardSocket.CAM_A,
    "CAM_B": dai.CameraBoardSocket.CAM_B,
    "CAM_C": dai.CameraBoardSocket.CAM_C,
    "CAM_D": dai.CameraBoardSocket.CAM_D,
}


def get_socket_from_name(
    name: str, name_to_socket: dict[str, str]
) -> dai.CameraBoardSocket:
    return socket_stringToCam[name_to_socket[name]]


def get_inv_R_T(R: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tmp = np.eye(4)
    tmp[:3, :3] = R
    tmp[:3, 3] = T
    inv_tmp = np.linalg.inv(tmp)
    inv_R = inv_tmp[:3, :3]
    inv_T = inv_tmp[:3, 3]

    return inv_R, inv_T
