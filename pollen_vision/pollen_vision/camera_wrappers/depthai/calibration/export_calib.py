"""Export calibration and two images from Depthai camera.

Provide raw data to be used for further image processing.

Example usage:

python pollen_vision/pollen_vision/camera_wrappers/depthai/calibration/export_calib.py \
     --config CONFIG_IMX296

generates the json file and two images in the current directory.

"""

import argparse
import logging
import time
from datetime import datetime

import cv2
from pollen_vision.camera_wrappers.depthai import SDKWrapper
from pollen_vision.camera_wrappers.depthai.utils import (
    get_config_file_path,
    get_config_files_names,
)

logging.basicConfig(level=logging.DEBUG)

valid_configs = get_config_files_names()

argParser = argparse.ArgumentParser(description="Export calibration and two images from Depthai camera.")

argParser.add_argument(
    "--config",
    type=str,
    required=True,
    choices=valid_configs,
    help=f"Configutation file name : {valid_configs}",
)

argParser.add_argument(
    "--export_path",
    help="Path to the exported data",
    default=".",
)

argParser.add_argument(
    "--save_calib_json",
    action="store_true",
    help="Export calibration json file",
)

argParser.add_argument(
    "--save_images",
    action="store_true",
    help="Export left and right raw images",
)

args = argParser.parse_args()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

w = SDKWrapper(get_config_file_path(args.config), rectify=False, jpeg_output=True)

if args.save_calib_json:
    w.save_calibration(f"{args.export_path}/calibration_{timestamp}.json")

if args.save_images:
    data, _, _ = w.get_data()
    # discard first image, luminosity is not stable yet
    time.sleep(1)
    data, _, _ = w.get_data()

    left_img = cv2.imdecode(data["left"], cv2.IMREAD_COLOR)
    right_img = cv2.imdecode(data["right"], cv2.IMREAD_COLOR)

    cv2.imwrite(f"{args.export_path}/left_raw_{timestamp}.jpg", left_img)
    cv2.imwrite(f"{args.export_path}/right_raw_{timestamp}.jpg", right_img)
