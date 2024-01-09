from importlib.resources import files
from typing import Any, Dict, List, Tuple

import cv2
import depthai as dai
import numpy as np
import numpy.typing as npt
from cv2 import aruco

socket_stringToCam = {
    "CAM_A": dai.CameraBoardSocket.CAM_A,
    "CAM_B": dai.CameraBoardSocket.CAM_B,
    "CAM_C": dai.CameraBoardSocket.CAM_C,
    "CAM_D": dai.CameraBoardSocket.CAM_D,
}


def get_socket_from_name(name: str, name_to_socket: Dict[str, str]) -> dai.CameraBoardSocket:
    return socket_stringToCam[name_to_socket[name]]


def get_inv_R_T(R: npt.NDArray[Any], T: npt.NDArray[Any]) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    tmp = np.eye(4)
    tmp[:3, :3] = R
    tmp[:3, 3] = T
    inv_tmp = np.linalg.inv(tmp)
    inv_R = inv_tmp[:3, :3]
    inv_T = inv_tmp[:3, 3]

    return inv_R, inv_T


def drawEpiLines(left: npt.NDArray[Any], right: npt.NDArray[Any], aruco_dict: aruco.Dictionary) -> npt.NDArray[Any]:
    concatIm = np.hstack((left, right))

    lcorners, lids, _ = aruco.detectMarkers(image=left, dictionary=aruco_dict)
    rcorners, rids, _ = aruco.detectMarkers(image=right, dictionary=aruco_dict)

    if len(lcorners) == 0 or len(rcorners) == 0:
        return concatIm

    lids = lids.reshape(-1)
    rids = rids.reshape(-1)

    lcorners_dict = {}
    for i in range(len(lids)):
        lid = lids[i]
        lcorners_dict[lid] = lcorners[i].reshape((-1, 2))

    rcorners_dict = {}
    for i in range(len(rids)):
        rid = rids[i]
        rcorners_dict[rid] = rcorners[i].reshape((-1, 2))

    avg_slope = []

    for lid in lids:
        if lid not in rids:
            continue

        lcorners = lcorners_dict[lid]
        rcorners = rcorners_dict[lid]
        for i in range(len(lcorners)):
            x1 = lcorners[i][0]
            y1 = lcorners[i][1]
            x2 = rcorners[i][0] + left.shape[1]
            y2 = rcorners[i][1]
            avg_slope.append(abs(y2 - y1))
            cv2.line(
                concatIm,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                1,
            )

    avg_slope_mean = np.mean(avg_slope)
    avg_slope_percent = avg_slope_mean / left.shape[0] * 100
    logging.info(f"AVG SLOPE : {np.round(avg_slope, 2)} px ({round(abs(avg_slope_percent), 2)} %)")
    return concatIm


def get_config_files_names() -> List[str]:
    path = files("config_files")
    return [file.stem for file in path.glob("**/*.json")]  # type: ignore[attr-defined]


def get_config_file_path(name: str) -> Any:
    path = files("config_files")
    for file in path.glob("**/*"):  # type: ignore[attr-defined]
        if file.stem == name:
            return file.resolve()
    return None
