"""This script is used to annotate a video with the OwlViT and SAM models.

The script takes a video as input and can output a new video with the detected objects and their segmentation masks.
Use the `--with-segmentation` flag to perform segmentation. By default, the script will only perform object detection.


Args:
    -v: the path to the video to annotate
    --with-segmentation: add this flag to perform segmentation
    -t: the detection threshold for the object detection model (default: 0.2)
    --classes: a list of classes to detect. Separate the classes with a space. Example: --classes 'robot' 'mug'

Example:
    python annotate_video.py -v path/to/video.mp4 --with-segmentation -t 0.2 --classes 'robot' 'mug'"

Output:
    The annotated video will be saved in the same directory as the input video with the suffix "_annotated".
"""

import argparse
import asyncio

import cv2
import numpy as np
import tqdm
from pollen_vision.utils import Annotator, get_bboxes
from pollen_vision.vision_models.object_detection import OwlVitWrapper
from pollen_vision.vision_models.object_segmentation import MobileSamWrapper
from recorder import Recorder

argParser = argparse.ArgumentParser(description="record sr")
argParser.add_argument(
    "-v",
    "--video",
    type=str,
    required=True,
    help="Video to annotate",
)
argParser.add_argument(
    "--with-segmentation",
    action="store_true",
    help="Whether to perform segmentation or not",
)
argParser.add_argument(
    "-t",
    "--threshold",
    type=float,
    default=0.2,
    help="Detection threshold for the object detection model",
)
argParser.add_argument(
    "--classes",
    nargs="+",
    required=True,
    help="Classes to detect. Separa Example: --classes 'robot' 'mug'",
)
args = argParser.parse_args()

cap_left = cv2.VideoCapture(args.video)
nb_frames_left = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))

print("Instantiating owl vit ...")
owl_vit = OwlVitWrapper()

use_segmentation = args.with_segmentation

if use_segmentation:
    print("Instantiating mobile sam ...")
    sam = MobileSamWrapper()

A = Annotator()


annotated_video_path = args.video.split(".")[0] + "_annotated.mp4"
rec_left = Recorder(annotated_video_path)

classes = args.classes
detection_threshold = args.threshold

print(f"Starting video annotation for classes: {classes} with detection threshold {detection_threshold}...")

for i in tqdm.tqdm(range(nb_frames_left)):
    ret, left_frame = cap_left.read()
    if not ret:
        break

    predictions = owl_vit.infer(
        im=left_frame,
        candidate_labels=classes,
        detection_threshold=detection_threshold,
    )
    bboxes = get_bboxes(predictions)

    if use_segmentation:
        masks = sam.infer(left_frame, bboxes)
    else:
        masks = []

    left_frame = A.annotate(im=left_frame, detection_predictions=predictions, masks=masks)

    asyncio.run(rec_left.new_im(left_frame.astype(np.uint8)))


print("Saving video ...")
rec_left.stop()
print("Video saved")
