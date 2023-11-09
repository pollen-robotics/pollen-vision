import cv2

from depthai_wrappers.cv_wrapper import CvWrapper

w = CvWrapper(
    "/home/antoine/Pollen/pollen-vision/config_files/CONFIG_IMX296.json",
    50,
    resize=(1280, 720),
    rectify=True,
)

while True:
    data, _, _ = w.get_data()
    cv2.imshow("left", data["left"])
    cv2.imshow("right", data["right"])
    cv2.waitKey(1)
