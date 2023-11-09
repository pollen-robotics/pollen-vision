import cv2
import numpy as np

from depthai_wrappers.depth_wrapper import DepthWrapper

w = DepthWrapper(
    "/home/antoine/Pollen/pollen-vision/config_files/CONFIG_CUSTOM_SR.json",
    60,
)

while True:
    data = w.get_data()
    cv2.imshow("left", data["left"])
    cv2.imshow("right", data["right"])
    cv2.imshow("depth", data["depth"])
    disparity = data["disparity"]

    disparity = data["disparity"]
    disparity = (disparity * (255 / w.depth_max_disparity)).astype(np.uint8)
    disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
    cv2.imshow("disparity", disparity)
    cv2.waitKey(1)
