import cv2
from pollen_vision.camera_wrappers.orbbec.orbbec_wrapper import OrbbecWrapper
from pollen_vision.utils.pcl_visualizer import PCLVisualizer

w = OrbbecWrapper()

K = w.get_K()
P = PCLVisualizer(K)

while True:
    data, _, _ = w.get_data()
    if "depth" not in data:
        continue
    depth = data["depth"]
    rgb = data["left"]

    P.update(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), depth)

    key = cv2.waitKey(1)
    P.tick()
