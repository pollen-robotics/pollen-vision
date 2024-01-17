import argparse

import cv2
import numpy as np

from depthai_wrappers.sdk_wrapper import SDKWrapper
from depthai_wrappers.utils import get_config_file_path, get_config_files_names

valid_configs = get_config_files_names()

argParser = argparse.ArgumentParser(description="depth wrapper example")
argParser.add_argument(
    "--config",
    type=str,
    required=True,
    choices=valid_configs,
    help=f"Configutation file name : {valid_configs}",
)
args = argParser.parse_args()
compute_depth = True
w = SDKWrapper(get_config_file_path(args.config), compute_depth=compute_depth, rectify=False)

cv2.namedWindow("disparity")

mouse_x, mouse_y = 0, 0


def callba(event, x, y, flags, param):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y


cv2.setMouseCallback("disparity", callba)

while True:
    data, _, _ = w.get_data()
    # cv2.imshow("left", data["left"])
    # cv2.imshow("right", data["right"])
    if compute_depth:
        # cv2.imshow("depthNode_left", data["depthNode_left"])
        # cv2.imshow("depthNode_right", data["depthNode_right"])
        # cv2.imshow("depth", data["depth"])
        disparity = data["disparity"]

        disparity = data["disparity"]
        disparity = (disparity * (255 / w.depth_max_disparity)).astype(np.uint8)
        disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)

        cv2.circle(disparity, (mouse_x, mouse_y), 5, (0, 0, 255), -1)
        cv2.imshow("disparity", disparity)

        depth_at_mouse = data["depth"][mouse_y, mouse_x]
        print("Depth at mouse: ", depth_at_mouse)

    cv2.waitKey(1)
