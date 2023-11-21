import argparse

import cv2
import numpy as np

from depthai_wrappers.depth_wrapper import DepthWrapper

argParser = argparse.ArgumentParser(description="Cv wrapper example")
argParser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to the configuration file.",
)
args = argParser.parse_args()

w = DepthWrapper(
    args.config,
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
