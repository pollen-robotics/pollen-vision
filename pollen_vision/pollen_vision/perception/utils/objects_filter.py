from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt


class ObjectsFilter:
    def __init__(self, max_objects_in_memory: int = 500) -> None:
        self.objects: List[Dict[str:Any]] = []  # type: ignore
        self.max_objects_in_memory = max_objects_in_memory
        self.pos_threshold = 0.05  # meters

    def tick(self) -> None:
        if len(self.objects) == 0:
            return
        to_remove = []
        # Also check objects that are at the same position
        for i in range(len(self.objects)):
            self.objects[i]["temporal_score"] = max(
                0, self.objects[i]["temporal_score"] - 0.05
            )  # TODO parametrize and tune this
            if self.objects[i]["temporal_score"] < 0.1:  # TODO parametrize and tune this
                # print(f"DEBUG FILTER: low score {self.objects[i]['temporal_score']}")
                to_remove.append(i)
            for j in range(i + 1, len(self.objects)):
                bbox1 = self.objects[i]["bbox"]
                bbox2 = self.objects[j]["bbox"]
                area_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                area_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                similar_bboxes = min(area_bbox1, area_bbox2) > 0.9 * max(area_bbox1, area_bbox2)

                if np.linalg.norm(self.objects[i]["pose"][:3, 3] - self.objects[j]["pose"][:3, 3]) < self.pos_threshold:
                    if self.objects[i]["name"] != self.objects[j]["name"] and similar_bboxes:
                        # objects are at the same position but have different names, and their bboxes of similar sizes
                        # probably the same object detected differently
                        # remove the one with the lowest detection score
                        if self.objects[i]["detection_score"] < self.objects[j]["detection_score"]:
                            to_remove.append(i)
                            # print(f"DEBUG FILTER: multiple objects at the same place")
                        else:
                            to_remove.append(j)
                            # print(f"DEBUG FILTER: multiple objects at the same place")

        # print(f"DEBUG FILTER, to remove: {to_remove}")
        self.objects = [self.objects[i] for i in range(len(self.objects)) if i not in to_remove]

    # pos : (x, y, z), bbox : [xmin, ymin, xmax, ymax]
    def push_observation(
        self,
        object_name: str,
        pose: npt.NDArray[np.float32],
        bbox: List[List],  # type: ignore
        mask: npt.NDArray[np.uint8],
        detection_score: float,
    ) -> None:
        if len(self.objects) == 0:
            self.objects.append(
                {
                    "name": object_name,
                    "pose": pose,
                    "temporal_score": 0.2,
                    "bbox": bbox,
                    "mask": mask,
                    "detection_score": detection_score,
                }
            )
            return

        if len(self.objects) > self.max_objects_in_memory:
            # print("wait a bit, too many objects in memory")
            return

        for i in range(len(self.objects)):
            if np.linalg.norm(pose[:3, 3] - self.objects[i]["pose"][:3, 3]) < self.pos_threshold:  # meters
                if object_name == self.objects[i]["name"]:  # merge same objects -> multiple observations of the same object
                    self.objects[i]["temporal_score"] = min(
                        1, self.objects[i]["temporal_score"] + 0.2
                    )  # TODO parametrize and tune this
                    new_pose = np.eye(4)
                    new_pose[:3, 3] = self.objects[i]["pose"][:3, 3] + 0.3 * (pose[:3, 3] - self.objects[i]["pose"][:3, 3])
                    self.objects[i]["pose"] = new_pose
                    self.objects[i]["bbox"] = bbox  # last bbox for now
                    self.objects[i]["mask"] = mask
                    self.objects[i]["detection_score"] = self.objects[i]["detection_score"] * 0.5 + detection_score * 0.5
                    return
            else:
                self.objects.append(
                    {
                        "name": object_name,
                        "pose": pose,
                        "temporal_score": 0.2,
                        "bbox": bbox,
                        "detection_score": detection_score,
                        "mask": mask,
                    }
                )

    def show_objects(self, threshold: float = 0.8) -> None:
        for i in range(len(self.objects)):
            if self.objects[i]["temporal_score"] > threshold:
                print(self.objects[i]["name"], self.objects[i]["pose"], self.objects[i]["temporal_score"])

    def get_objects(self, threshold: float = 0.0) -> List[Dict]:  # type: ignore
        tmp = sorted(self.objects.copy(), key=lambda k: np.linalg.norm(k["pose"][:3, 3]))
        return [obj for obj in tmp if obj["temporal_score"] > threshold]
