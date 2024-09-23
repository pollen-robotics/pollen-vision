import cv2
from pollen_vision.camera_wrappers.depthai import SDKWrapper
from pollen_vision.camera_wrappers.depthai.utils import get_config_file_path
from pollen_vision.utils import Annotator
from pollen_vision.vision_models.object_detection import YoloWorldWrapper

w = SDKWrapper(get_config_file_path("CONFIG_SR"))
A = Annotator()
Y = YoloWorldWrapper()

while True:
    data, _, _ = w.get_data()
    image = data["left"]

    predictions = Y.infer(
        image,
        candidate_labels=[
            "mug",
            "orange sunglasses",
            "blue plate",
            "dark plate",
            "orange fruit",
            "apple",
            "green tape roll",
            "metallic food can",
        ],
        detection_threshold=0.05,
    )

    annotated = A.annotate(image, predictions)
    cv2.imshow("annotated", annotated)
    cv2.waitKey(1)
