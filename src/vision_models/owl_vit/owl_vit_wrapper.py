from typing import Dict, List

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image, ImageDraw
from transformers import pipeline


class OwlVitWrapper:
    def __init__(self) -> None:
        self.checkpoint = "google/owlvit-base-patch32"
        self.detector = pipeline(model=self.checkpoint, task="zero-shot-object-detection", device=torch.cuda.current_device())

    def infer(self, im: npt.NDArray[np.uint8], candidate_labels: List[str]) -> List[Dict]:
        im = Image.fromarray(im)
        predictions: List[Dict] = self.detector(im, candidate_labels=candidate_labels)
        return predictions

    def draw_predictions(self, im: npt.NDArray[np.uint8], predictions: Dict) -> Image:
        im: Image = Image.fromarray(im)
        draw = ImageDraw.Draw(im)
        for prediction in predictions:
            box = prediction["box"]
            label = prediction["label"]
            score = prediction["score"]
            xmin, ymin, xmax, ymax = box.values()
            draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
            draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="white")

        return im

    def get_bboxes(self, predictions: List[Dict]) -> List[List]:
        bboxes = []
        for prediction in predictions:
            box = prediction["box"]
            xmin, ymin, xmax, ymax = box.values()
            bboxes.append([xmin, ymin, xmax, ymax])

        return bboxes

    def get_labels(self, predictions: List[Dict]) -> List[str]:
        labels = []
        for prediction in predictions:
            labels.append(prediction["label"])

        return labels
