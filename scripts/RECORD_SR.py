import argparse
import asyncio
import datetime
import os
import time

import numpy as np
from pollen_vision.camera_wrappers.depthai import SDKWrapper
from pollen_vision.camera_wrappers.depthai.utils import get_config_file_path
from recorder import FPS, Recorder

argParser = argparse.ArgumentParser(description="record sr")
argParser.add_argument(
    "-o",
    "--out",
    type=str,
    required=True,
    help="Output directory",
)
argParser.add_argument(
    "-n",
    "--name",
    type=str,
    required=False,
    default=str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    help="name of the session",
)
args = argParser.parse_args()

os.makedirs(args.out, exist_ok=True)
session_path = os.path.join(args.out, args.name)
os.makedirs(session_path, exist_ok=True)

w = SDKWrapper(get_config_file_path("CONFIG_SR"), compute_depth=True, fps=FPS)

rec_left = Recorder(os.path.join(session_path, "left_video.mp4"))
rec_right = Recorder(os.path.join(session_path, "right_video.mp4"))
rec_depth = Recorder(os.path.join(session_path, "depth_video.mp4"))

rec_left.start()
rec_right.start()
rec_depth.start()

print("")
print("")
print("========================")
print("Starting to record, press ctrl + c to stop")
try:
    while True:
        start = time.time()
        data, _, ts = w.get_data()

        left_img = data["left"]
        right_img = data["right"]
        depth = data["depth"]
        depth = np.repeat(depth[:, :, np.newaxis], 3, axis=2)

        asyncio.run(rec_left.new_im(left_img.astype(np.uint8)))
        asyncio.run(rec_right.new_im(right_img.astype(np.uint8)))
        asyncio.run(rec_depth.new_im(depth.astype(np.uint8)))
        end = time.time()

        took = end - start

        time.sleep(max(0, (1 / FPS) - took))  # Compensating for jitteriness

except KeyboardInterrupt:
    print("Saving ...")
    pass

rec_left.stop()
rec_right.stop()
rec_depth.stop()

print("Done!")
print("Done!")
