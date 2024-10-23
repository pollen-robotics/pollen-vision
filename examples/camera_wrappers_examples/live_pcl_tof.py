import argparse

import cv2
from pollen_vision.camera_wrappers import TOFWrapper
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

w = TOFWrapper(get_config_file_path(args.config), crop=False, fps=30, create_pointcloud=True)

P = PCLVisualizer()
P.add_frame("origin")


while True:
    data, lat, _ = w.get_data()

    rgb = data["left"]

    P.update(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), points=data["pointcloud"])

    key = cv2.waitKey(1)
    P.tick()
