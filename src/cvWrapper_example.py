import argparse

import cv2

from depthai_wrappers.cv_wrapper import CvWrapper
from depthai_wrappers.utils import get_config_file_path, get_config_files_names

valid_configs = get_config_files_names()

argParser = argparse.ArgumentParser(description="Cv wrapper example")
argParser.add_argument(
    "--config",
    type=str,
    required=True,
    choices=valid_configs,
    help=f"Configutation file name : {valid_configs}",
)
args = argParser.parse_args()

w = CvWrapper(
    get_config_file_path(args.config),
    50,
    resize=(1280, 720),
    rectify=True,
)

while True:
    data, _, _ = w.get_data()
    cv2.imshow("left", data["left"])
    cv2.imshow("right", data["right"])
    cv2.waitKey(1)
