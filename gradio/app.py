"""
Gradio app for pollen-vision

This script creates a Gradio app for pollen-vision. The app allows users to perform object detection and object segmentation using the OWL-ViT and MobileSAM models.
"""

from datasets import load_dataset
import gradio as gr

import numpy as np
import numpy.typing as npt
from typing import Any, Dict, List

from pollen_vision.vision_models.object_detection import OwlVitWrapper
from pollen_vision.vision_models.object_segmentation import MobileSamWrapper
from pollen_vision.vision_models.utils import Annotator, get_bboxes


owl_vit = OwlVitWrapper()
mobile_sam = MobileSamWrapper()
annotator = Annotator()


def object_detection(img: npt.NDArray[np.uint8], text_queries: List[str], score_threshold: float) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = owl_vit.infer(
        im=img, candidate_labels=text_queries, detection_threshold=score_threshold
    )
    return predictions


def object_segmentation(
    img: npt.NDArray[np.uint8], object_detection_predictions: List[Dict[str, Any]]
) -> List[npt.NDArray[np.uint8]]:
    bboxes = get_bboxes(predictions=object_detection_predictions)
    masks: List[npt.NDArray[np.uint8]] = mobile_sam.infer(im=img, bboxes=bboxes)
    return masks


def query(
    task: str,
    img: npt.NDArray[np.uint8],
    text_queries: List[str],
    score_threshold: float,
) -> npt.NDArray[np.uint8]:
    object_detection_predictions = object_detection(img=img, text_queries=text_queries, score_threshold=score_threshold)

    if task == "Object detection + segmentation (OWL-ViT + MobileSAM)":
        masks = object_segmentation(img=img, object_detection_predictions=object_detection_predictions)
        img = annotator.annotate(im=img, detection_predictions=object_detection_predictions, masks=masks)
        return img

    img = annotator.annotate(im=img, detection_predictions=object_detection_predictions)
    return img


description = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam nec purus et nunc tincidunt tincidunt.
"""

demo_inputs = [
    gr.Dropdown(
        [
            "Object detection (OWL-ViT)",
            "Object detection + segmentation (OWL-ViT + MobileSAM)",
        ],
        label="Choose a task",
        value="Object detection (OWL-ViT)",
    ),
    gr.Image(),
    "text",
    gr.Slider(0, 1, value=0.1),
]

rdt_dataset = load_dataset("pollen-robotics/reachy-doing-things", split="train")

img_kitchen_detection = rdt_dataset[11]["image"]
img_kitchen_segmentation = rdt_dataset[12]["image"]

demo_examples = [
    [
        "Object detection (OWL-ViT)",
        img_kitchen_detection,
        ["kettle", "black mug", "sink", "blue mug", "sponge", "bag of chips"],
        0.15,
    ],
    [
        "Object detection + segmentation (OWL-ViT + MobileSAM)",
        img_kitchen_segmentation,
        ["blue mug", "paper cup", "kettle", "sponge"],
        0.12,
    ],
]

demo = gr.Interface(
    fn=query,
    inputs=demo_inputs,
    outputs="image",
    title="pollen-vision",
    description=description,
    examples=demo_examples,
)
demo.launch()
