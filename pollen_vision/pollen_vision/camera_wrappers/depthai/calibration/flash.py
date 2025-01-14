import argparse
import logging

from pollen_vision.camera_wrappers.depthai import SDKWrapper
from pollen_vision.camera_wrappers.depthai.utils import (
    get_config_file_path,
    get_config_files_names,
)

logging.basicConfig(level=logging.DEBUG)

valid_configs = get_config_files_names()

argParser = argparse.ArgumentParser(description="Flash calibration parameters found by multical to the EEPROM of the device.")
argParser.add_argument(
    "--config",
    type=str,
    required=True,
    choices=valid_configs,
    help=f"Configutation file name : {valid_configs}",
)
argParser.add_argument(
    "--calib_json_file",
    required=True,
    help="Path to the calibration json file",
)
argParser.add_argument("--tof", action="store_true", help="Has tof sensor ?")
args = argParser.parse_args()

w = SDKWrapper(get_config_file_path(args.config))
w.flash(args.calib_json_file, tof=args.tof)
