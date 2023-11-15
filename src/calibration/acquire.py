import argparse
import os

import cv2
import numpy as np

from depthai_wrappers.cv_wrapper import CvWrapper

argParser = argparse.ArgumentParser(
    description="Acquire images from a luxonis camera and save them to disk."
)
argParser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to the configuration file.",
)
argParser.add_argument(
    "--imagesPath",
    type=str,
    default="./calib_images/",
    help="Directory where the acquired images are stored (default ./calib_images/)",
)
args = argParser.parse_args()

w = CvWrapper(args.config, rectify=False)

left_path = os.path.join(args.imagesPath, "left")
right_path = os.path.join(args.imagesPath, "right")
os.makedirs(left_path, exist_ok=True)
os.makedirs(right_path, exist_ok=True)
i = 0
while True:
    data, _, _ = w.get_data()
    _data = {}
    for name in data.keys():
        _data[name] = data[name]

    concat = np.hstack((_data["left"], _data["right"]))
    cv2.imshow(name, cv2.resize(concat, (0, 0), fx=0.5, fy=0.5))
    key = cv2.waitKey(1)
    if key == 13:
        cv2.imwrite(os.path.join(left_path, str(i) + ".png"), data["left"])
        cv2.imwrite(os.path.join(right_path, str(i) + ".png"), data["right"])
        print("Saved image pair ", i)
        i += 1
