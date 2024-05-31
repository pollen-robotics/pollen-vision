from typing import Dict, List

import cv2
import numpy as np
import numpy.typing as npt

from .labels import Labels
from .utils import get_bboxes, get_labels, get_scores


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
