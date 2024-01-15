import cv2
import numpy as np

from depthai_wrappers.sdk_wrapper import SDKWrapper
from depthai_wrappers.utils import get_config_file_path

sr = SDKWrapper(get_config_file_path("CONFIG_SR"), compute_depth=True, rectify=False, mx_id="18443010B1FC5D1200")
head = SDKWrapper(get_config_file_path("CONFIG_IMX296"), compute_depth=False, rectify=True, mx_id="194430108167641300")


while True:
    data_sr, _, _ = sr.get_data()
    data_head, _, _ = head.get_data()
    cv2.imshow("sr_left", data_sr["left"])
    cv2.imshow("sr_right", data_sr["right"])
    cv2.imshow("sr_depthNode_left", data_sr["depthNode_left"])
    cv2.imshow("sr_depthNode_right", data_sr["depthNode_right"])
    cv2.imshow("depth", data_sr["depth"])

    disparity = data_sr["disparity"]
    disparity = data_sr["disparity"]
    disparity = (disparity * (255 / sr.depth_max_disparity)).astype(np.uint8)
    disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
    cv2.imshow("disparity", disparity)

    cv2.imshow("left_head", data_head["left"])
    cv2.imshow("right_head", data_head["right"])
    cv2.waitKey(1)
