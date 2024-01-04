import argparse

import cv2
import numpy as np

from depthai_wrappers.depth_wrapper import DepthWrapper
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

w = DepthWrapper(
    get_config_file_path(args.config),
    50,
)

while True:
    data, _, _ = w.get_data()
    cv2.imshow("left", data["left"])
    cv2.imshow("right", data["right"])
    cv2.imshow("depth", data["depth"])
    disparity = data["disparity"]

    disparity = data["disparity"]
    disparity = (disparity * (255 / w.depth_max_disparity)).astype(np.uint8)
    disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
    cv2.imshow("disparity", disparity)
    cv2.waitKey(1)
