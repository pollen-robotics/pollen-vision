import logging

import cv2
from cv2 import aruco
from pollen_vision.camera_wrappers import BaslerWrapper
from pollen_vision.camera_wrappers.depthai.utils import drawEpiLines

logging.basicConfig(level=logging.DEBUG)


ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)

w = BaslerWrapper(undistort=True, calib_file_path="../calibration/calib_images/calibration.json")

while True:
    try:
        data, _, _ = w.get_data()
    except Exception as e:
        continue

    _data = {}
    for name in data.keys():
        _data[name] = data[name]
    epi = drawEpiLines(_data["left"], _data["right"], ARUCO_DICT)
    epi = cv2.resize(epi, (0, 0), fx=0.4, fy=0.4)
    cv2.imshow("epi", epi)
    key = cv2.waitKey(1)

    if key == 27 or key == ord("q"):
        break
