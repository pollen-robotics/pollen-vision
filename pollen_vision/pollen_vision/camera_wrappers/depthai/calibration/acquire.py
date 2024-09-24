import argparse
import os

import cv2
import numpy as np
from pollen_vision.camera_wrappers.depthai import SDKWrapper, TOFWrapper
from pollen_vision.camera_wrappers.depthai.utils import (
    get_config_file_path,
    get_config_files_names,
)

valid_configs = get_config_files_names()
argParser = argparse.ArgumentParser(description="Acquire images from a luxonis camera and save them to disk.")
argParser.add_argument(
    "--config",
    type=str,
    required=True,
    choices=valid_configs,
    help=f"Configutation file name : {valid_configs}",
)
argParser.add_argument(
    "--imagesPath",
    type=str,
    default="./calib_images/",
    help="Directory where the acquired images are stored (default ./calib_images/)",
)
argParser.add_argument("--tof", action="store_true", help="Has tof sensor ?")
args = argParser.parse_args()

if not args.tof:
    w = SDKWrapper(get_config_file_path(args.config), compute_depth=False, rectify=False)
else:
    w = TOFWrapper(get_config_file_path("CONFIG_IMX296_TOF"), fps=30)


left_path = os.path.join(args.imagesPath, "left")
right_path = os.path.join(args.imagesPath, "right")
os.makedirs(left_path, exist_ok=True)
os.makedirs(right_path, exist_ok=True)
if args.tof:
    tof_path = os.path.join(args.imagesPath, "tof")
    os.makedirs(tof_path, exist_ok=True)

print("Press return to save an image pair.")
print("(Keep the focus on the opencv window for the inputs to register.)")
print("Press esc or q to exit.")

i = 0
while True:
    data, _, _ = w.get_data()
    _data = {}
    for name in data.keys():
        _data[name] = data[name]

    if args.tof:
        tof_amplitude = _data["tof_amplitude"]
        tof_amplitude = cv2.resize(tof_amplitude, _data["left"].shape[:2][::-1])
        tof_amplitude = np.dstack((tof_amplitude, tof_amplitude, tof_amplitude))
        concat = np.hstack((_data["left"], _data["right"], tof_amplitude))
    else:
        concat = np.hstack((_data["left"], _data["right"]))
    cv2.imshow("concat", cv2.resize(concat, (0, 0), fx=0.5, fy=0.5))
    key = cv2.waitKey(1)
    if key == 13:
        cv2.imwrite(os.path.join(left_path, str(i) + ".png"), data["left"])
        cv2.imwrite(os.path.join(right_path, str(i) + ".png"), data["right"])
        if args.tof:
            cv2.imwrite(os.path.join(tof_path, str(i) + ".png"), data["tof_amplitude"])
        print("Saved image pair ", i, "to", left_path, " and ", right_path)
        i += 1
    elif key == 27 or key == ord("q"):
        break
