import argparse

import cv2
import numpy as np
from pollen_vision.camera_wrappers.depthai import SDKWrapper
from pollen_vision.camera_wrappers.depthai.utils import (
    get_config_file_path,
    get_config_files_names,
)
from pollen_vision.perception.utils.pcl_visualizer import PCLVisualizer

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

K = w.cam_config.get_K_left()
P = PCLVisualizer(K)

while True:
    data, lat, _ = w.get_data()
    print(lat["depthNode_left"], lat["depthNode_right"])

    depth = data["depth"]
    rgb = data["left"]
    disparity = data["disparity"]
    P.update(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), depth)

    disparity = (disparity * (255 / w.depth_max_disparity)).astype(np.uint8)
    disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
    cv2.imshow("disparity", disparity)
    cv2.imshow("left", data["depthNode_left"])
    cv2.imshow("right", data["depthNode_right"])

    key = cv2.waitKey(1)
    P.tick()
