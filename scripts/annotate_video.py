import argparse
import asyncio

import cv2
import numpy as np
import tqdm
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
args = argParser.parse_args()

cap_left = cv2.VideoCapture(args.video)
# cap_right = cv2.VideoCapture("test/bouteille/right_video.mp4")
# cap_depth = cv2.VideoCapture("test/bouteille/depth_video.mp4")
nb_frames_left = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))

print("Instantiating owl vit ...")
owl_vit = OwlVitWrapper()
print("Instantiating mobile sam ...")
sam = MobileSamWrapper()


annotated_video_path = args.video.split(".")[0] + "_annotated.mp4"
rec_left = Recorder(annotated_video_path)

print("Starting")

for i in tqdm.tqdm(range(nb_frames_left)):
    ret, left_frame = cap_left.read()
    # ret, right_frame = cap_right.read()
    # ret, depth_frame = cap_depth.read()
    if not ret:
        break
    # depth_frame = depth_frame[:, :, 0]

    predictions = owl_vit.infer(
        left_frame,
        ["croissant pastry", "napkin"],
    )

    bboxes = owl_vit.get_bboxes(predictions)
    labels = owl_vit.get_labels(predictions)
    masks = sam.infer(left_frame, bboxes)

    left_frame = owl_vit.draw_predictions(left_frame, predictions)
    left_frame = sam.annotate(np.array(left_frame), masks, bboxes, labels, labels_colors=owl_vit.labels_colors)

    asyncio.run(rec_left.new_im(left_frame.astype(np.uint8)))


print("Saving video ...")
rec_left.stop()
print("Video saved")
