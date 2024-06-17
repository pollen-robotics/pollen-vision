from typing import List

import numpy as np
import numpy.typing as npt
import torch
from mobile_sam import SamPredictor, sam_model_registry
from pollen_vision.perception.utils import get_checkpoint_path, get_checkpoints_names


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

    def infer(
        self, im: npt.NDArray[np.uint8], bboxes: List[List[np.uint8]] = [], points_list: List[List[List[np.uint8]]] = []
    ) -> List[npt.NDArray[np.uint8]]:
        """Returns a list of masks found in the input image.
        A mask is a binary image where the pixels inside the mask are set to 1 and the pixels outside the mask are set to 0.

        Args:
            - im: the input image (opencv image, numpy array)
            - bboxes: a list of bounding boxes in the format [[xmin, ymin, xmax, ymax], ...]
            - points: a list of list of points. One list of points per object to segment
                For example, [[[x1, y1], [x2, y2], ...], [[x1, y1], [x2, y2], ...]]
                                       obj1                       obj2
        """

        if len(bboxes) == 0 and len(points_list) == 0:
            return []

        self._predictor.set_image(np.array(im))

        _masks = []
        for bbox in bboxes:
            _mask, _, _ = self._predictor.predict(box=np.array(bbox))
            _masks.append(_mask)

        for points in points_list:
            _mask, _, _ = self._predictor.predict(point_coords=np.array(points), point_labels=[1] * len(points))
            _masks.append(_mask)

        masks: List = []  # type: ignore
        for i in range(len(_masks)):
            m = np.array(_masks[i]).astype(np.uint8)
            m = m[0, :, :]
            m.swapaxes(0, 1)
            masks.append(m)

        return masks
