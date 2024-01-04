import argparse

from depthai_wrappers.cv_wrapper import CvWrapper

argParser = argparse.ArgumentParser(description="Flash calibration parameters found by multical to the EEPROM of the device.")
argParser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to the configuration file.",
)
argParser.add_argument(
    "--calib_json_file",
    required=True,
    help="Path to the calibration json file",
)
args = argParser.parse_args()
w = CvWrapper(args.config)
w.flash(args.calib_json_file)
