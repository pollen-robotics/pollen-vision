import argparse

import cv2
import numpy as np
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

w = TOFWrapper(get_config_file_path(args.config), crop=False)

K = w.get_K()
P = PCLVisualizer(K)
P.add_frame("origin")


def colorizeDepth(frameDepth):
    invalidMask = frameDepth == 0
    # Log the depth, minDepth and maxDepth
    try:
        minDepth = np.percentile(frameDepth[frameDepth != 0], 3)
        maxDepth = np.percentile(frameDepth[frameDepth != 0], 95)
        logDepth = np.log(frameDepth, where=frameDepth != 0)
        logMinDepth = np.log(minDepth)
        logMaxDepth = np.log(maxDepth)
        np.nan_to_num(logDepth, copy=False, nan=logMinDepth)
        # Clip the values to be in the 0-255 range
        logDepth = np.clip(logDepth, logMinDepth, logMaxDepth)

        # Interpolate only valid logDepth values, setting the rest based on the mask
        depthFrameColor = np.interp(logDepth, (logMinDepth, logMaxDepth), (0, 255))
        depthFrameColor = np.nan_to_num(depthFrameColor)
        depthFrameColor = depthFrameColor.astype(np.uint8)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
        # Set invalid depth pixels to black
        depthFrameColor[invalidMask] = 0
    except IndexError:
        # Frame is likely empty
        depthFrameColor = np.zeros((frameDepth.shape[0], frameDepth.shape[1], 3), dtype=np.uint8)
    except Exception as e:
        raise e
    return depthFrameColor


mouse_x, mouse_y = 0, 0


def cv_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y


# cv2.namedWindow("depth")
# cv2.setMouseCallback("depth", cv_callback)
while True:
    data, lat, _ = w.get_data()
    # print(lat["depthNode_left"], lat["depthNode_right"])

    depth = data["depth"]
    rgb = data["left"]

    # colorized_depth = colorizeDepth(depth)
    # disparity = data["disparity"]

    P.update(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), depth)

    # disparity = (disparity * (255 / w.depth_max_disparity)).astype(np.uint8)
    # disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
    # cv2.imshow("disparity", disparity)
    # cv2.imshow("left", data["depthNode_left"])
    # cv2.imshow("right", data["depthNode_right"])
    # cv2.imshow("depth", colorized_depth)
    # cv2.imshow("rgb", rgb)
    # print(depth[mouse_y, mouse_x])

    key = cv2.waitKey(1)
    P.tick()
