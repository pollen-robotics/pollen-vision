import argparse

import cv2

from depthai_wrappers.cv_wrapper import CvWrapper

argParser = argparse.ArgumentParser(description="Cv wrapper example")
argParser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to the configuration file.",
)
args = argParser.parse_args()


w = CvWrapper(
    args.config,
    50,
    resize=(1280, 720),
    rectify=True,
)

while True:
    data, _, _ = w.get_data()
    cv2.imshow("left", data["left"])
    cv2.imshow("right", data["right"])
    cv2.waitKey(1)
