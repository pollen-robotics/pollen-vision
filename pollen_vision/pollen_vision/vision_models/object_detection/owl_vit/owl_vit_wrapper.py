from typing import Dict, List

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from transformers import pipeline


class OwlVitWrapper:
    """A wrapper for the OwlVit model."""

    def __init__(self) -> None:
        self._checkpoint = "google/owlvit-base-patch32"
        self._device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        self._detector = pipeline(model=self._checkpoint, task="zero-shot-object-detection", device=self._device)

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
            - "box": the bounding box of the object, in the format [xmin, ymin, xmax, ymax]
        """
        im = Image.fromarray(im)
        predictions: List[Dict] = self._detector(im, candidate_labels=candidate_labels)  # type: ignore
        predictions = [prediction for prediction in predictions if prediction["score"] > detection_threshold]
        return predictions
