import cv2
import numpy as np
import depthai as dai
from pollen_vision.camera_wrappers import TOFWrapper
from pollen_vision.camera_wrappers.depthai.utils import get_config_file_path


t = TOFWrapper(get_config_file_path("CONFIG_AR0234_TOF"), fps=60)


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


# P = PCLVisualizer(t.get_K())
cv2.namedWindow("depth")
cv2.setMouseCallback("depth", cv_callback)
print(dai.__version__)
while True:
    data, _, _ = t.get_data()
    left = data["left"]
    # right = data["right"]
    # left = cv2.resize(left, (640, 480))
    # right = cv2.resize(right, (640, 480))
    depth = data["depth"]
    colorized_depth = colorizeDepth(depth)
    cv2.imshow("left", left)
    # cv2.imshow("right", right)
    print(data["depth"][mouse_y, mouse_x])
    # colorized_depth = cv2.circle(depth, (mouse_x, mouse_y), 5, (0, 255, 0), -1)
    blended = cv2.addWeighted(left, 0.5, colorized_depth, 0.5, 0)
    cv2.imshow("blended", blended)
    cv2.imshow("depth", depth)
    # P.update(cv2.cvtColor(left, cv2.COLOR_BGR2RGB), depth)
    # P.tick()
    cv2.waitKey(1)
