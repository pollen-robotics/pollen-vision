import argparse

import cv2
from pollen_vision.camera_wrappers import SDKWrapper
from pollen_vision.camera_wrappers.depthai.utils import (
    get_config_file_path,
    get_config_files_names,
)
from pollen_vision.utils.pcl_visualizer import PCLVisualizer

valid_configs = get_config_files_names()

argParser = argparse.ArgumentParser(description="depth wrapper example")
argParser.add_argument(
    "--config",
    type=str,
    required=True,
    choices=valid_configs,
    help=f"Configutation file name : {valid_configs}",
)
args = argParser.parse_args()

w = SDKWrapper(get_config_file_path(args.config), compute_depth=True)

K = w.get_K()
P = PCLVisualizer(K)

while True:
    data, lat, _ = w.get_data()

    depth = data["depth"]
    rgb = data["left"]

    P.update(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), depth)

    cv2.imshow("depth", depth)
    cv2.imshow("rgb", rgb)

    key = cv2.waitKey(1)
    P.tick()
