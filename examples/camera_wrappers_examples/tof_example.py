import cv2
import depthai as dai
from pollen_vision.camera_wrappers import TOFWrapper
from pollen_vision.camera_wrappers.depthai.utils import (
    colorizeDepth,
    get_config_file_path,
)

t = TOFWrapper(get_config_file_path("CONFIG_IMX296_TOF"), fps=30, crop=False)

print(dai.__version__)
while True:
    data, _, _ = t.get_data()
    left = data["left"]

    depth = data["depth"]
    tof_intensity = data["tof_intensity"]

    cv2.imshow("left", left)
    cv2.imshow("tof intensity", tof_intensity)
    colorized_depth = colorizeDepth(data["depth"])
    blended = cv2.addWeighted(left, 0.5, colorized_depth, 0.5, 0)
    cv2.imshow("blended", blended)
    cv2.imshow("colorized_depth", colorized_depth)
    cv2.waitKey(1)
