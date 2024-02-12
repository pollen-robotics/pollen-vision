from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image, ImageDraw
from transformers import pipeline


def random_color() -> Tuple[int, int, int]:
return tuple(np.random.randint(0, 255, 3))


class OwlVitWrapper:
    def __init__(self) -> None:
        self.checkpoint = "google/owlvit-base-patch32"
        self.detector = pipeline(model=self.checkpoint, task="zero-shot-object-detection", device=torch.cuda.current_device())

    def infer(self, im: npt.NDArray[np.uint8], candidate_labels: List[str]) -> List[Dict]:  # type: ignore
        im = Image.fromarray(im)
        predictions: List[Dict] = self.detector(im, candidate_labels=candidate_labels)  # type: ignore
        return predictions

    def draw_predictions(self, in_im: npt.NDArray[np.uint8], predictions: Dict) -> Image:  # type: ignore
        im: Image = Image.fromarray(in_im)
        draw = ImageDraw.Draw(im)

        # Pick one random color per class label
        colors_per_label: Dict[str, Tuple[int, int, int]] = {}

        for prediction in predictions:
            box = prediction["box"]
            label = prediction["label"]

            if label not in colors_per_label.keys():
                colors_per_label[label] = random_color()

            score = prediction["score"]
            xmin, ymin, xmax, ymax = box.values()
            draw.rectangle((xmin, ymin, xmax, ymax), outline=colors_per_label[label], width=1)
            draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="white")

        return im

    def get_bboxes(self, predictions: List[Dict]) -> List[List]:  # type: ignore
        bboxes = []
        for prediction in predictions:
            box = prediction["box"]
            xmin, ymin, xmax, ymax = box.values()
            bboxes.append([xmin, ymin, xmax, ymax])

        return bboxes

    def get_labels(self, predictions: List[Dict]) -> List[str]:  # type: ignore
        labels = []
        for prediction in predictions:
            labels.append(prediction["label"])

        return labels
