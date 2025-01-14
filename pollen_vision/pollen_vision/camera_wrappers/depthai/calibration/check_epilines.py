import argparse
import logging

import cv2
import numpy as np
from cv2 import aruco
from pollen_vision.camera_wrappers.depthai import SDKWrapper, TOFWrapper
from pollen_vision.camera_wrappers.depthai.utils import (
    drawEpiLines,
    get_config_file_path,
    get_config_files_names,
)

logging.basicConfig(level=logging.DEBUG)

valid_configs = get_config_files_names()
argParser = argparse.ArgumentParser(description="Check that the stereo calibration is correct.")
argParser.add_argument(
    "--config",
    type=str,
    required=True,
    choices=valid_configs,
    help=f"Configutation file name : {valid_configs}",
)
argParser.add_argument("--tof", action="store_true", help="Has tof sensor ?")
args = argParser.parse_args()

ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)

if not args.tof:
    w = SDKWrapper(get_config_file_path(args.config), rectify=True, resize=(1280, 720), fps=60)
else:
    w = TOFWrapper(get_config_file_path("CONFIG_IMX296_TOF"), fps=30, rectify=True)


while True:
    data, _, _ = w.get_data()

    _data = {}
    for name in data.keys():
        _data[name] = data[name]
    epi = drawEpiLines(_data["left"], _data["right"], ARUCO_DICT)
    epi = cv2.resize(epi, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("epi", epi)

    if args.tof:
        right_resized = cv2.resize(_data["right"], _data["tof_intensity"].shape[:2][::-1])
        tof_im = _data["tof_intensity"]
        tof_im = np.dstack((tof_im, tof_im, tof_im))
        epi_right_tof = drawEpiLines(right_resized, tof_im, ARUCO_DICT)
        cv2.imshow("epi_right_tof", epi_right_tof)
    key = cv2.waitKey(1)

    if key == 27 or key == ord("q"):
        break
