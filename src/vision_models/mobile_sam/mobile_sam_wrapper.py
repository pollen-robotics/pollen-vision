from typing import Dict, List, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import torch
from mobile_sam import SamPredictor, sam_model_registry

from vision_models.utils import get_checkpoint_path, get_checkpoints_names


class MobileSamWrapper:
    """A wrapper for the MobileSam model."""

    def __init__(self, checkpoint_name: str = "mobile_sam") -> None:
        """
        Args:
            - checkpoint_name: the name of the checkpoint to load
        """
        valid_names = get_checkpoints_names()
        assert checkpoint_name in valid_names
        self._checkpoint_path = get_checkpoint_path(checkpoint_name)

        self._model_type = "vit_t"
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        mobile_sam = sam_model_registry[self._model_type](checkpoint=self._checkpoint_path)

        mobile_sam.to(device=self._device)
        mobile_sam.eval()

        self._predictor = SamPredictor(mobile_sam)

    def infer(self, im: npt.NDArray[np.uint8], bboxes: List[List]) -> List:  # type: ignore
        """Returns a list of masks found in the input image.
        A mask is a binary image where the pixels inside the mask are set to 1 and the pixels outside the mask are set to 0.
        """
        self._predictor.set_image(np.array(im))

        _masks = []
        for bbox in bboxes:
            _mask, _, _ = self._predictor.predict(box=np.array(bbox))
            _masks.append(_mask)

        masks: List = []  # type: ignore
        for i in range(len(_masks)):
            m = np.array(_masks[i]).astype(np.uint8)
            m = m[0, :, :]
            m.swapaxes(0, 1)
            masks.append(m)

        return masks

    def annotate(
        self,
        im: npt.NDArray[np.uint8],
        masks: List[npt.NDArray[np.uint8]],
        bboxes: List[List[int]],
        labels: List[str],
        labels_colors: Dict[str, Tuple[int, int, int]] = {},
    ) -> npt.NDArray[np.uint8]:
        """Draws the masks and labels on top of the input image and returns the annotated image.
        Args:
            - im: the input image
            - masks: a list of masks
            - bboxes: a list of bounding boxes
            - labels: a list of labels
            - labels_colors: a dictionary of colors for each label (if not set, mask will be drawn in white)
        """
        im = np.array(im)
        for i in range(len(masks)):
            mask = masks[i]
            label = labels[i]

            # Draw transparent color mask on top of im
            label = labels[i]
            color = (255, 255, 255) if label not in labels_colors else labels_colors[label]
            overlay = np.zeros_like(im, dtype=np.uint8)
            overlay[mask != 0] = color
            overlay[mask == 0] = im[mask == 0]
            im = cv2.addWeighted(overlay, 0.5, im, 1 - 0.5, 0)

            # Write label at x, y position in im
            # bbox = bboxes[i]
            # x, y = bbox[0], bbox[1]
            # im = cv2.putText(im, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        return im
