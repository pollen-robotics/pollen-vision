from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image, ImageDraw
from transformers import pipeline

from vision_models.utils import random_color


class OwlVitWrapper:
    """A wrapper for the OwlVit model."""

    def __init__(self) -> None:
        self._checkpoint = "google/owlvit-base-patch32"
        self._device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        self._detector = pipeline(model=self._checkpoint, task="zero-shot-object-detection", device=self._device)
        self.labels_colors: Dict[str, Tuple[int, int, int]] = {}

    def infer(
        self,
        im: npt.NDArray[np.uint8],
        candidate_labels: List[str],
        detection_threshold: float = 0.0,
    ) -> List[Dict]:  # type: ignore
        """Returns a list of predictions found in the input image.
        Args:
            - im: the input image
            - candidate_labels: a list of candidate labels
            - detection_threshold: the detection threshold to filter out predictions with a score below this threshold

        Returns a list of predictions found in the input image.

        A prediction is a dictionary with the following keys:
            - "label": the label of the object
            - "score": the score of the prediction
            - "box": the bounding box of the object
        """
        im = Image.fromarray(im)
        predictions: List[Dict] = self._detector(im, candidate_labels=candidate_labels)  # type: ignore
        predictions = [prediction for prediction in predictions if prediction["score"] > detection_threshold]
        return predictions

    def draw_predictions(self, in_im: npt.NDArray[np.uint8], predictions: List[Dict]) -> Image:  # type: ignore
        """Draws the predictions on a copy of the input image and returns the annotated image."""

        im: Image = Image.fromarray(in_im)
        draw = ImageDraw.Draw(im)

        for prediction in predictions:
            box = prediction["box"]
            label = prediction["label"]

            if label not in self.labels_colors.keys():
                self.labels_colors[label] = random_color()

            score = prediction["score"]
            xmin, ymin, xmax, ymax = box.values()
            draw.rectangle((xmin, ymin, xmax, ymax), outline=self.labels_colors[label], width=5)
            # draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="black", font_size=20)

        return im

    def get_bboxes(self, predictions: List[Dict]) -> List[List]:  # type: ignore
        """Returns a list of bounding boxes from the predictions."""
        bboxes = []
        for prediction in predictions:
            box = prediction["box"]
            xmin, ymin, xmax, ymax = box.values()
            bboxes.append([xmin, ymin, xmax, ymax])

        return bboxes

    def get_labels(self, predictions: List[Dict]) -> List[str]:  # type: ignore
        """Returns a list of labels from the predictions."""
        labels = []
        for prediction in predictions:
            labels.append(prediction["label"])

        return labels
