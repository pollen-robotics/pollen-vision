import argparse
import os
import threading
import time

from moves import get_board, replay_recording
from reachy_sdk import ReachySDK

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
argParser.add_argument(
    "--skipAcquire",
    action="store_true",
)
args = argParser.parse_args()

left_path = os.path.join(args.imagesPath, "left")
right_path = os.path.join(args.imagesPath, "right")
os.makedirs(left_path, exist_ok=True)
os.makedirs(right_path, exist_ok=True)

reachy = ReachySDK(host="192.168.1.252")
if not args.skipAcquire:
    w = CvWrapper(args.config, rectify=False)
else:
    w = None


# reachy.turn_off_smoothly("r_arm")
# reachy.turn_off_smoothly("head")
# time.sleep(1)
# exit()
reachy.turn_on("r_arm")
reachy.turn_on("head")

if not args.skipAcquire:
    board_acquired = get_board(reachy)
    time.sleep(3)
    if not board_acquired:
        reachy.turn_off_smoothly("r_arm")
        time.sleep(3)
        exit()

    print("Procedding to acquiring images")
    replay_recording(
        reachy,
        "./recording_acquire.pkl",
        w,
        left_path,
        right_path,
        acquire=True,
        duration=1.5,
    )
    print("Done acquiring images")
    time.sleep(1)
print("Computing calibration ...")

calib_thread = threading.Thread(
    target=lambda: os.system(
        f"multical calibrate --image_path ~/Pollen/pollen-vision/src/calibration/auto_calib/calib_images/ --boards ~/Pollen/multical/example_boards/pollen_charuco.yaml --isFisheye True"
    )
)
calib_thread.start()
while calib_thread.is_alive():
    replay_recording(
        reachy,
        "./recording_think.pkl",
        w,
        left_path,
        right_path,
        acquire=False,
        duration=0.5,
    )
calib_thread.join()
# TODO check RMS here ?
print("Flashing calibration ...")
if not args.skipAcquire:
    w.close()
w = CvWrapper(args.config, rectify=True, resize=(1280, 720), fps=50)
success = w.flash(os.path.join(args.imagesPath, "calibration.json"))
if not success:
    reachy.turn_off_smoothly("r_arm")
    reachy.turn_off("head")
    print("Failed to flash calibration, exiting")
    exit()

print("Checking epilines ...")
replay_recording(
    reachy,
    "./recording_check_epilines.pkl",
    w,
    left_path,
    right_path,
    acquire=False,
    check_epilines=True,
    duration=3.0,
)

print("HAPPY")
replay_recording(
    reachy,
    "./recording_win.pkl",
    w,
    left_path,
    right_path,
    acquire=False,
    duration=0.15,
)
reachy.turn_off_smoothly("r_arm")
reachy.turn_off("head")
