from typing import Dict, List

import numpy as np
import numpy.typing as npt
from inference.models import YOLOWorld


class YoloWorldWrapper:
    """A wrapper for the YOLO World model."""

    def __init__(self) -> None:
        self._model = YOLOWorld(model_id="yolo_world/l")
        self._classes: List[str] = []

    def infer(
        self, im: npt.NDArray[np.uint8], candidate_labels: List[str], detection_threshold: float = 0.1
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
        if candidate_labels != self._classes:
            self.set_classes(candidate_labels)

        results = self._model.infer(im, confidence=detection_threshold)
        preds = []
        for res in results:
            if "predictions" in res:
                preds = list(res[1])

        predictions = []
        for pred in preds:
            prediction = {}
            prediction["label"] = self._classes[pred.class_id]
            prediction["score"] = pred.confidence
            box = {}
            box["xmin"] = int(pred.x - pred.width / 2)
            box["ymin"] = int(pred.y - pred.height / 2)
            box["xmax"] = int(pred.x + pred.width / 2)
            box["ymax"] = int(pred.y + pred.height / 2)
            prediction["box"] = box
            predictions.append(prediction)

        return predictions

    def set_classes(self, classes: List[str]) -> None:
        self._classes = classes
        self._model.set_classes(self._classes)
