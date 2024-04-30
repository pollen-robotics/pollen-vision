"""A collection of utility functions for the vieion models wrappers"""

from importlib.resources import files
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import numpy.typing as npt


def get_centroid(mask: npt.NDArray[np.uint8]) -> Tuple[int, int]:
    x_center, y_center = np.argwhere(mask == 1).sum(0) / np.count_nonzero(mask)
    return int(y_center), int(x_center)


def uv_to_xyz(z: float, u: float, v: float, K: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    cx = K[0, 2]
    cy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    # Computing position in camera frame
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.array([x, y, z])


# Actually only computes the position ignoring rotation for now. Still returns pose matrix, with rotation set to identity
def get_object_pose_in_world(
    depth: npt.NDArray[np.float32],
    mask: npt.NDArray[np.uint8],
    T_world_cam: npt.NDArray[np.float32],
    K: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    u, v = get_centroid(mask)

    d = depth.copy()
    d[mask == 0] = 0
    average_depth = d[d != 0].mean()

    xyz = uv_to_xyz(average_depth * 0.1, u, v, K)
    xyz *= 0.01

    T_cam_object = np.eye(4)  # Rotation is identity for now
    T_cam_object[:3, 3] = xyz
    T_world_object = T_world_cam @ T_cam_object
    T_world_object[:3, :3] = np.eye(3)

    return T_world_object


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
                to_remove.append(i)
            for j in range(i + 1, len(self.objects)):
                if np.linalg.norm(self.objects[i]["pos"] - self.objects[j]["pos"]) < self.pos_threshold:
                    if self.objects[i]["name"] != self.objects[j]["name"]:
                        # objects are at the same position but have different names
                        # remove the one with the lowest detection score
                        if self.objects[i]["detection_score"] < self.objects[j]["detection_score"]:
                            to_remove.append(i)
                        else:
                            to_remove.append(j)

        self.objects = [self.objects[i] for i in range(len(self.objects)) if i not in to_remove]

    # pos : (x, y, z), bbox : [xmin, ymin, xmax, ymax]
    def push_observation(
        self,
        object_name: str,
        pos: npt.NDArray[np.float32],
        bbox: List[List],
        mask: npt.NDArray[np.uint8],
        detection_score: float,
    ) -> None:  # type: ignore
        if len(self.objects) == 0:
            self.objects.append(
                {
                    "name": object_name,
                    "pos": pos,
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
            if np.linalg.norm(pos - self.objects[i]["pos"]) < self.pos_threshold:  # meters
                if object_name == self.objects[i]["name"]:  # merge same objects -> multiple observations of the same object
                    self.objects[i]["temporal_score"] = min(
                        1, self.objects[i]["temporal_score"] + 0.2
                    )  # TODO parametrize and tune this
                    self.objects[i]["pos"] = self.objects[i]["pos"] + 0.3 * (pos - self.objects[i]["pos"])
                    self.objects[i]["bbox"] = bbox  # last bbox for now
                    self.objects[i]["mask"] = mask
                    self.objects[i]["detection_score"] = self.objects[i]["detection_score"] * 0.5 + detection_score * 0.5
                    return
            else:
                self.objects.append(
                    {"name": object_name, "pos": pos, "temporal_score": 0.2, "bbox": bbox, "detection_score": detection_score}
                )

    def show_objects(self, threshold: float = 0.8) -> None:
        for i in range(len(self.objects)):
            if self.objects[i]["temporal_score"] > threshold:
                print(self.objects[i]["name"], self.objects[i]["pos"], self.objects[i]["temporal_score"])

    def get_objects(self, threshold: float = 0.8) -> List[Dict]:  # type: ignore
        tmp = sorted(self.objects, key=lambda k: np.linalg.norm(k["pos"]))
        return [obj for obj in tmp if obj["temporal_score"] > threshold]


class Labels:
    """A class to store the labels and their colors."""

    def __init__(self) -> None:
        self.labels: Dict[str, Tuple[int, int, int]] = {}
        self.labels[""] = (255, 255, 255)

    def push(self, labels: List[str], colors: List[Tuple[int, int, int]] = []) -> None:
        """Pushes a list of labels and associated color to the main labels dictionary.

        If the color list is not set, a random color will be assigned to each label not already in the dictionnary.
        """
        if colors != []:
            if len(colors) != len(labels):
                raise ValueError("The length of the labels and colors lists must be the same.")

        for label in labels:
            if label not in self.labels:
                if colors != []:
                    self.labels[label] = colors[labels.index(label)]
                else:
                    self.labels[label] = random_color()

    def get_color(self, label: str) -> Tuple[int, int, int]:
        """Returns the color of the label."""
        return self.labels[label]


class Annotator:
    """A class to annotate images with bounding boxes, masks and labels.
    This class handles the persistance of the colors across frames.

    By default, the colors are randomly assigned to the labels.
    You can also set the colors manually by passing an instance of the Labels class to the constructor.

    At any time, you can edit/push the colors of the labels by accessing the self.labels public attribute from outside
    """

    def __init__(self, labels: Labels = Labels()) -> None:
        self.labels = labels

    def annotate(
        self,
        im: npt.NDArray[np.uint8],
        detection_predictions: List[Dict] = [],  # type: ignore
        masks: List[npt.NDArray[np.uint8]] = [],
    ) -> npt.NDArray[np.uint8]:
        """Draws the masks and labels on top of the input image and returns the annotated image.

        masks arg is optional. If not set, masks will not be drawn.

        Args:
            - im: the input image
            - detection_predictions: a dictionary containing the predictions of the object detection model. Keys are :
                - "box": a dictionary containing the bounding box coordinates (xmin, ymin, xmax, ymax)
                - "label": the label of the object
                - "score": the confidence score of the prediction
            - masks: a list of masks to draw on top of the image
        """
        im = np.array(im)

        bboxes = get_bboxes(detection_predictions)
        pred_labels = get_labels(detection_predictions)
        scores = get_scores(detection_predictions)

        len_masks = len(masks)
        len_bboxes = len(bboxes)
        len_labels = len(pred_labels)
        len_scores = len(scores)

        for length in [len_masks, len_bboxes, len_labels, len_scores]:
            assert length in [0, max(len_masks, len_bboxes, len_labels, len_scores)]

        self.labels.push(pred_labels)

        for i in range(max(len_masks, len_bboxes, len_labels, len_scores)):
            mask = masks[i] if len(masks) > 0 else np.zeros_like(im)
            bbox = bboxes[i] if len(bboxes) > 0 else [0, 0, 0, 0]
            label = pred_labels[i] if len(pred_labels) > 0 else ""
            score = str(np.round(scores[i], 2)) if len(scores) > 0 else ""

            # Draw transparent color mask on top of im
            color = self.labels.get_color(label)
            color = np.array(color).astype(np.uint8).tolist()

            overlay = np.zeros_like(im, dtype=np.uint8)

            if len(masks) > 0:
                overlay[mask != 0] = color
                overlay[mask == 0] = im[mask == 0]
                im = np.array(cv2.addWeighted(overlay, 0.5, im, 1 - 0.5, 0))

            im = np.array(cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2))

            # Write label at x, y position in im
            x, y = bbox[0], bbox[1]
            im = np.array(cv2.putText(im, label + " " + score, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA))

        return im


def get_bboxes(predictions: List[Dict]) -> List[List]:  # type: ignore
    """Returns a list of bounding boxes from the predictions."""
    bboxes = []
    for prediction in predictions:
        box = prediction["box"]
        xmin, ymin, xmax, ymax = box.values()
        bboxes.append([xmin, ymin, xmax, ymax])

    return bboxes


def get_checkpoints_names() -> List[str]:
    """Returns the names of the checkpoints available in the checkpoints directory."""
    path = files("checkpoints")
    names = []
    for file in path.glob("**/*.pt"):  # type: ignore[attr-defined]
        names.append(file.stem)

    for file in path.glob("**/*.pth"):  # type: ignore[attr-defined]
        names.append(file.stem)

    return names


def get_checkpoint_path(name: str) -> Any:
    """Returns the path of the checkpoint based on its name."""
    path = files("checkpoints")
    for file in path.glob("**/*"):  # type: ignore[attr-defined]
        if file.stem == name:
            return str(file.resolve())
    return None


def get_labels(predictions: List[Dict]) -> List[str]:  # type: ignore
    """Returns a list of labels from the predictions."""
    labels = []
    for prediction in predictions:
        labels.append(prediction["label"])

    return labels


def get_scores(predictions: List[Dict]) -> List[float]:  # type: ignore
    """Returns a list of scores from the predictions."""
    scores = []
    for prediction in predictions:
        scores.append(prediction["score"])

    return scores


def random_color() -> Tuple[int, int, int]:
    """Returns a random color."""
    return tuple(np.random.randint(0, 255, 3))
