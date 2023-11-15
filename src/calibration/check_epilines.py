import argparse

import cv2
from cv2 import aruco

from depthai_wrappers.cv_wrapper import CvWrapper
from depthai_wrappers.utils import drawEpiLines

argParser = argparse.ArgumentParser(
    description="Check that the stereo calibration is correct."
)
argParser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to the configuration file.",
)
args = argParser.parse_args()

ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)

CHARUCO_BOARD = aruco.CharucoBoard(
    (11, 8),
    squareLength=0.022,
    markerLength=0.0167,
    dictionary=ARUCO_DICT,
)
CHARUCO_BOARD.setLegacyPattern(True)
w = CvWrapper(args.config, rectify=True)

while True:
    data, _, _ = w.get_data()
    _data = {}
    for name in data.keys():
        _data[name] = data[name]
    epi = drawEpiLines(_data["left"], _data["right"], ARUCO_DICT)
    epi = cv2.resize(epi, (0, 0), fx=0.9, fy=0.9)
    cv2.imshow("epi", epi)
    cv2.waitKey(1)
