import argparse
import os

import depthai as dai

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--calib_file",
    type=str,
    required=True,
    help="calibration backup file",
)
args = parser.parse_args()

ret = input("This will erase currently flashed calibration. Continue ?(y/n)")
while ret not in ["y", "n"]:
    print("Invalid input. Try again.")
    ret = input("This will erase currently flashed calibration. Continue ?(y/n)")

if ret == "n":
    exit()

os.environ["DEPTHAI_ALLOW_FACTORY_FLASHING"] = "235539980"
with dai.Device(dai.OpenVINO.VERSION_UNIVERSAL, dai.UsbSpeed.HIGH) as device:
    calibData = dai.CalibrationHandler(args.calib_file)

    try:
        device.flashCalibration2(calibData)
        print("Successfully flashed calibration")
    except Exception as ex:
        print(f"Failed flashing calibration: {ex}")
