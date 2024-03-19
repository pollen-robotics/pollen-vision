"""
Gradio app for pollen-vision

This script creates a Gradio app for pollen-vision. The app allows users to perform object detection and object
segmentation using the OWL-ViT and MobileSAM models.
"""

from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt
from datasets import load_dataset
from pollen_vision.vision_models.object_detection import OwlVitWrapper
from pollen_vision.vision_models.object_segmentation import MobileSamWrapper
from pollen_vision.vision_models.utils import Annotator, get_bboxes

import gradio as gr

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
Welcome to the demo of pollen-vision, a simple and unified Python library to zero-shot computer vision models curated
for robotics use cases. **Pollen-vision** is designed for ease of installation and use, composed of independent modules
that can be combined to create a 3D object detection pipeline, getting the position of the objects in 3D space (x, y, z).

\n\nIn this demo, you have the option to choose between two tasks: object detection and object detection + segmentation.
The models available are:

- **OWL-VIT** (Open World Localization - Vision Transformer, By Google Research): this model performs text-conditionned
zero-shot 2D object localization in RGB images.
- **Mobile SAM**: A lightweight version of the Segment Anything Model (SAM) by Meta AI. SAM is a zero shot image
segmentation model. It can be prompted with bounding boxes or points. (https://github.com/ChaoningZhang/MobileSAM)

\n\nYou can input images in this demo in three ways: either by trying out the provided examples, by uploading an image
of your choice, or by capturing an image from your computer's webcam.
Additionally, you should provide text queries representing a list of objects to detect. Separate each object with a comma.
The last input parameter is the detection threshold (ranging from 0 to 1), which defaults to 0.1.

\n\nCheck out our blog post introducing pollen-vision or its <a href="https://github.com/pollen-robotics/pollen-vision">
Github repository</a> for more info!
"""

demo_inputs = [
    gr.Dropdown(  # type: ignore
        [
            "Object detection (OWL-ViT)",
            "Object detection + segmentation (OWL-ViT + MobileSAM)",
        ],
        label="Choose a task",
        value="Object detection (OWL-ViT)",
    ),
    gr.Image(),  # type: ignore
    "text",
    gr.Slider(0, 1, value=0.1),  # type: ignore
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

demo = gr.Interface(  # type: ignore
    fn=query,
    inputs=demo_inputs,
    outputs="image",
    title="Use zero-shot computer vision models with pollen-vision",
    description=description,
    examples=demo_examples,
)
demo.launch()
