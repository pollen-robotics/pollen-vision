from typing import List

import cv2
import numpy as np
import numpy.typing as npt
import torch
from mobile_sam import SamPredictor, sam_model_registry

from vision_models.utils import get_checkpoint_path, get_checkpoints_names


class MobileSamWrapper:
    def __init__(self, checkpoint_name: str = "mobile_sam") -> None:
        valid_names = get_checkpoints_names()
        assert checkpoint_name in valid_names
        self.checkpoint_path = get_checkpoint_path(checkpoint_name)

        self.model_type = "vit_t"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        mobile_sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)

        mobile_sam.to(device=self.device)
        mobile_sam.eval()

        self.predictor = SamPredictor(mobile_sam)

    def infer(self, im: npt.NDArray[np.uint8], bboxes: List[List]) -> None:
        self.predictor.set_image(np.array(im))

        _masks = []
        for bbox in bboxes:
            _mask, _, _ = self.predictor.predict(box=np.array(bbox))
            _masks.append(_mask)

        masks = []
        for i in range(len(_masks)):
            m = np.array(_masks[i]).astype(np.uint8)
            m = m[0, :, :]
            m.swapaxes(0, 1)
            masks.append(m)

        return masks

    def annotate(self, im, masks, bboxes, labels):
        for i in range(len(masks)):
            mask = masks[i]
            bbox = bboxes[i]
            x, y = bbox[0], bbox[1]
            label = labels[i]
            im[mask == 1] = [255, 255, 255]

            # Write label at x, y position in im
            im = cv2.putText(im, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return im
