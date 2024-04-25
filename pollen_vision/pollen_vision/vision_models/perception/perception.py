import threading
import time
from typing import Dict, List, Optional

import cv2
import FramesViewer.utils as fv_utils
import numpy as np
import numpy.typing as npt
from pollen_vision.camera_wrappers import CameraWrapper
from pollen_vision.camera_wrappers.depthai import SDKWrapper
from pollen_vision.camera_wrappers.depthai.utils import get_config_file_path
from pollen_vision.vision_models.object_detection import OwlVitWrapper
from pollen_vision.vision_models.object_segmentation import MobileSamWrapper
from pollen_vision.vision_models.utils import (
    Annotator,
    ObjectsFilter,
    get_bboxes,
    get_labels,
    get_object_pose_in_world,
)


class Perception:
    # camera_wrapper : An implementation of the CameraWrapper abstract class
    # T_world_cam : Transformation matrix from camera to world (pose of the camera expressed in the world frame)
    # freq : update frequency in Hz
    def __init__(self, camera_wrapper: CameraWrapper, T_world_cam: npt.NDArray[np.float32], freq: float = 1.0) -> None:
        self.cam = camera_wrapper
        self.T_world_cam = T_world_cam
        self.freq = freq
        self.OWL = OwlVitWrapper()
        self.SAM = MobileSamWrapper()
        self.A = Annotator()
        self.OF = ObjectsFilter()

        self.tracked_objects: list[str] = []
        self.last_im = None
        self.last_depth = None
        self.last_predictions: List[Dict] = []  # type: ignore
        self.last_masks: List[npt.NDArray[np.uint8]] = []

        self._lastTick = time.time()

    def start(self, visualize: bool = False) -> None:
        self._t = threading.Thread(target=self.tick, name="perception thread")
        self._t.daemon = True
        self.visualize = visualize
        self._lastTick = time.time()
        self._t.start()

    def tick(self) -> None:
        while True:
            elapsed = time.time() - self._lastTick
            if elapsed < 1 / self.freq:
                time.sleep(1 / self.freq - elapsed)

            data, _, _ = self.cam.get_data()
            self.last_im = data["left"]
            self.last_depth = data["depth"]

            self.OF.tick()
            if self.visualize and self.last_im is not None:
                objs = self.get_objects()
                annotated = self.last_im.copy()
                for obj in objs:
                    pos2D = (int((obj["bbox"][0] + obj["bbox"][2]) / 2), int((obj["bbox"][1] + obj["bbox"][3]) / 2))
                    annotated = cv2.circle(self.last_im, pos2D, 5, (0, 255, 0), -1)
                cv2.imshow("annotated", annotated)
                cv2.waitKey(1)

            if len(self.tracked_objects) == 0:
                self._lastTick = time.time()  # Lame
                continue

            self.last_predictions = self.OWL.infer(
                cv2.cvtColor(self.last_im, cv2.COLOR_BGR2RGB), self.tracked_objects, detection_threshold=0.1
            )

            if len(self.last_predictions) == 0:
                self._lastTick = time.time()  # Lame
                continue

            bboxes = get_bboxes(self.last_predictions)
            labels = get_labels(self.last_predictions)
            self.last_masks = self.SAM.infer(self.last_im, bboxes=bboxes)

            for i, mask in enumerate(self.last_masks):
                T_world_object = get_object_pose_in_world(self.last_depth, mask, self.T_world_cam, self.cam.get_K())
                self.OF.push_observation(labels[i], T_world_object, bboxes[i], mask)

            self._lastTick = time.time()

    def get_objects(self) -> List[Dict]:  # type: ignore
        """
        Return list of filtered objects sorted by distance.
        """
        return self.OF.get_objects()  # type: ignore

    def get_object_info(self, object_name: str) -> Dict:  # type: ignore
        """
        Return the object info for a given object name.
        """
        if object_name not in self.tracked_objects:
            return {}

        detected_objects = [obj["name"] for obj in self.get_objects()]
        if object_name not in detected_objects:
            return {}

        # position, rgb, mask, depth
        info = {"name": object_name, "position": np.eye(4), "rgb": self.last_im, "mask": None, "depth": self.last_depth}

        for obj in self.get_objects():
            if obj["name"] == object_name:
                info["position"] = self.get_object_position(object_name)
                info["mask"] = obj["mask"]
                break

        return info

    def set_tracked_objects(self, objects: list[str]) -> None:
        for obj in objects:
            if obj not in self.tracked_objects:
                print(f"Adding tracking for object: {obj}")
                self.tracked_objects.append(obj)

    def get_object_position(self, object_name: str) -> Optional[npt.NDArray[np.float64]]:
        """
        Return the position of the object in the world frame.
        """
        if object_name not in self.tracked_objects:
            return None

        for obj in self.get_objects():
            if obj["name"] == object_name:
                obj_pos: npt.NDArray[np.float64] = obj["pos"]
                return obj_pos

        return None


if __name__ == "__main__":
    S = SDKWrapper(get_config_file_path("CONFIG_SR"), compute_depth=True)
    T_world_cam = fv_utils.make_pose([0.03, -0.15, 0.1], [0, 0, 0])
    perception = Perception(S, T_world_cam, freq=10)
    perception.set_tracked_objects(["orange sunglasses"])
    perception.start(visualize=True)

    while True:
        print("==")
        print(perception.get_objects())
        print("==")
        time.sleep(0.1)
