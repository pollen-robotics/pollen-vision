"""A collection of utility functions for the vieion models wrappers"""

from importlib.resources import files
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import numpy.typing as npt


class Labels:
    """A class to store the labels and their colors."""

    def __init__(self) -> None:
        self.labels: Dict[str, Tuple[int, int, int]] = {}

    def push(self, labels: List[str]) -> None:
        """Pushes a label and its color to the labels dictionary."""
        for label in labels:
            if label not in self.labels:
                self.labels[label] = random_color()

    def get_color(self, label: str) -> Tuple[int, int, int]:
        """Returns the color of the label."""
        return self.labels[label]

    def get_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Returns the colors of the labels."""
        return self.labels


def annotate(
    im: npt.NDArray[np.uint8],
    masks: List[npt.NDArray[np.uint8]] = [],
    bboxes: List[List[int]] = [],
    labels: List[str] = [],
    scores: List[float] = [],
    labels_colors: Dict[str, Tuple[int, int, int]] = {},
) -> npt.NDArray[np.uint8]:
    """Draws the masks and labels on top of the input image and returns the annotated image.

    All the args are optional, except for the image. If not set, the corresponding element will not be drawn.
    Args:
        - im: the input image
        - masks: a list of masks
        - bboxes: a list of bounding boxes
        - labels: a list of labels
        - scores: a list of scores
        - labels_colors: a dictionary of colors for each label (if not set, mask will be drawn in white)
    """
    im = np.array(im)
    for i in range(len(masks)):
        mask = masks[i] if len(masks) > 0 else np.zeros_like(im)
        bbox = bboxes[i] if len(bboxes) > 0 else [0, 0, 0, 0]
        label = labels[i] if len(labels) > 0 else ""
        score = str(np.round(scores[i], 2)) if len(scores) > 0 else ""

        # Draw transparent color mask on top of im
        color = (255, 255, 255) if label not in labels_colors else labels_colors[label]
        color = np.array(color).astype(np.uint8).tolist()

        overlay = np.zeros_like(im, dtype=np.uint8)
        overlay[mask != 0] = color
        overlay[mask == 0] = im[mask == 0]
        im = cv2.addWeighted(overlay, 0.5, im, 1 - 0.5, 0)

        im = cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # Write label at x, y position in im
        x, y = bbox[0], bbox[1]
        im = cv2.putText(im, label + " " + score, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    return im


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


def random_color() -> Tuple[int, int, int]:
    """Returns a random color."""
    return tuple(np.random.randint(0, 255, 3))
