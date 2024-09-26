import threading
import time
from typing import Dict, List, Optional

import cv2
import FramesViewer.utils as fv_utils
import numpy as np
import numpy.typing as npt
from pollen_vision.camera_wrappers import CameraWrapper
from pollen_vision.camera_wrappers.pollen_sdk_camera.pollen_sdk_camera_wrapper import (
    PollenSDKCameraWrapper,
)
from pollen_vision.utils import (
    Annotator,
    ObjectsFilter,
    get_bboxes,
    get_labels,
    get_object_pose_in_world,
    get_scores,
)
from pollen_vision.vision_models.object_detection import OwlVitWrapper, YoloWorldWrapper
from pollen_vision.vision_models.object_segmentation import MobileSamWrapper


class Perception:
    # camera_wrapper : An implementation of the CameraWrapper abstract class
    # T_world_cam : Transformation matrix from camera to world (pose of the camera expressed in the world frame)
    # freq : update frequency in Hz
    def __init__(
        self,
        camera_wrapper: CameraWrapper,
        T_world_cam: Optional[npt.NDArray[np.float32]] = None,
        freq: float = 1.0,
        yolo_thres: float = 0.01,
    ) -> None:
        self.cam = camera_wrapper
        if isinstance(self.cam, PollenSDKCameraWrapper) and self.cam.cam_name == "teleop":
            self.cam_name = self.cam.cam_name
            head_rotation_matrix = self.cam.get_head_orientation().rotation_matrix
            T_world_neck = np.eye(4)
            T_world_neck[:3, :3] = head_rotation_matrix
            T_world_neck[:3, 3] = [0.0, 0.0, 0.125]
            T_neck_cam = fv_utils.make_pose([0.035, 0.033, 0.056], [-90, 0, -90])
            self.T_world_cam = T_world_neck @ T_neck_cam
        elif isinstance(self.cam, PollenSDKCameraWrapper) and self.cam.cam_name == "depth":
            self.cam_name = self.cam.cam_name
            T_world_cam = fv_utils.make_pose([0.077051, 0.011, 0.024908], [-90, 0, -90])
            self.T_world_cam = fv_utils.rotateInSelf(T_world_cam, [-48, 0, 0])
        else:
            self.cam_name = "manual"
            self.T_world_cam = T_world_cam

        self.freq = freq
        self.Owl = OwlVitWrapper()
        self.YOLO = YoloWorldWrapper()
        self.SAM = MobileSamWrapper()
        self.A = Annotator()
        self.OF = ObjectsFilter()

        self.tracked_objects: list[str] = []
        self.last_im = None
        self.last_depth = None
        self.last_predictions: List[Dict] = []  # type: ignore
        self.last_masks: List[npt.NDArray[np.uint8]] = []
        self._yolo_thres = yolo_thres
        self._lastTick = time.time()

    def start(self, visualize: bool = False) -> None:
        self.run_thread = True
        self._t = threading.Thread(target=self.tick, name="perception thread")
        self._t.daemon = True
        self.visualize = visualize
        self._lastTick = time.time()
        self._t.start()

    def stop(self) -> None:
        self.run_thread = False

    def tick(self) -> None:  # noqa: C901
        while self.run_thread:
            elapsed = time.time() - self._lastTick
            if elapsed < 1 / self.freq:
                time.sleep(0.00001)
                continue

            data, _, _ = self.cam.get_data()
            self.last_im = data["left"]
            self.last_depth = data["depth"]

            self.OF.tick()
            if self.visualize and self.last_im is not None:
                objs = self.get_objects_infos()
                annotated = self.last_im.copy()
                for obj in objs:
                    pos2D = (int((obj["bbox"][0] + obj["bbox"][2]) / 2), int((obj["bbox"][1] + obj["bbox"][3]) / 2))
                    annotated = cv2.circle(annotated, pos2D, 5, (0, 255, 0), -1)

            if len(self.tracked_objects) == 0:
                self._lastTick = time.time()  # Lame
                continue

            self.last_predictions = self.YOLO.infer(self.last_im, self.tracked_objects, detection_threshold=self._yolo_thres)
            if len(self.last_predictions) == 0:
                self._lastTick = time.time()  # Lame
                continue

            bboxes = get_bboxes(self.last_predictions)
            labels = get_labels(self.last_predictions)
            scores = get_scores(self.last_predictions)
            self.last_masks = self.SAM.infer(self.last_im, bboxes=bboxes)

            if self.visualize and self.last_im is not None:
                annotated = self.A.annotate(annotated, self.last_predictions, self.last_masks)
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                cv2.imshow("annotated", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    self.stop()

            if self.cam_name == "teleop":
                head_rotation_matrix = self.cam.get_head_orientation().rotation_matrix
                T_world_neck = np.eye(4)
                T_world_neck[:3, :3] = head_rotation_matrix
                T_world_neck[:3, 3] = [0.0, 0.0, 0.125]
                T_neck_cam = fv_utils.make_pose([0.035, 0.033, 0.056], [-90, 0, -90])
                self.T_world_cam = T_world_neck @ T_neck_cam

            for i, mask in enumerate(self.last_masks):
                T_world_object = get_object_pose_in_world(self.last_depth, mask, self.T_world_cam, self.cam.get_K())
                self.OF.push_observation(labels[i], T_world_object, bboxes[i], mask, scores[i])

            self._lastTick = time.time()

    def get_objects_infos(self, threshold: float = 0.8) -> List[Dict]:  # type: ignore
        """
        Return list of filtered objects sorted by distance.
        """
        objects_infos = []
        objects = self.OF.get_objects(threshold=threshold)
        for obj in objects:
            info = {
                "name": obj["name"],
                "pose": obj["pose"],
                "rgb": self.last_im,
                "mask": obj["mask"],
                "depth": self.last_depth,
                "bbox": obj["bbox"],
                "temporal_score": obj["temporal_score"],
                "detection_score": obj["detection_score"],
            }
            objects_infos.append(info)

        return objects_infos

    def set_tracked_objects(self, objects: list[str]) -> None:
        for obj in objects:
            if obj not in self.tracked_objects:
                # print(f"Adding tracking for object: {obj}")
                self.tracked_objects.append(obj)


if __name__ == "__main__":
    from pollen_vision.camera_wrappers.depthai import SDKWrapper
    from pollen_vision.camera_wrappers.depthai.utils import get_config_file_path

    # from pollen_vision.camera_wrappers.pollen_sdk_camera.pollen_sdk_camera_wrapper import (
    #     PollenSDKCameraWrapper,
    # )
    # from reachy2_sdk import ReachySDK

    T_world_cam = fv_utils.make_pose([0.03, -0.15, 0.1], [0, 0, 0])

    S = SDKWrapper(get_config_file_path("CONFIG_SR"), compute_depth=True)
    # reachy = ReachySDK(host="localhost")
    # S = PollenSDKCameraWrapper(reachy)

    perception = Perception(S, T_world_cam, freq=30)
    perception.set_tracked_objects(["mug", "grey duct tape", "pen"])
    perception.start(visualize=True)

    while True:
        print("==")

        objs = perception.get_objects_infos()
        for obj in objs:
            print(obj["name"], np.linalg.norm(obj["pose"][:3, 3]), obj["temporal_score"], obj["detection_score"])
        for i in perception.last_predictions:
            print(f"raw: {i}")
        print("==")
        time.sleep(0.1)
