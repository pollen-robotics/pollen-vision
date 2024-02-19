import argparse
import asyncio
import time

import cv2
import numpy as np
from recorder import Recorder

from vision_models.mobile_sam.mobile_sam_wrapper import MobileSamWrapper
from vision_models.owl_vit.owl_vit_wrapper import OwlVitWrapper

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
i = -1
mean_processing_time = 0
processing_times = []
while True:
    i += 1

    # if i > 45:
    #     break

    start = time.time()
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

    end = time.time()
    took = end - start
    processing_times.append(took)
    processing_times = processing_times[-10:]
    mean_processing_time = np.mean(np.array(processing_times))

    print(
        "[" + str(i) + "/" + str(nb_frames_left) + "] Estimated remaining time:",
        round((mean_processing_time * (nb_frames_left - i)) / 60, 2),
        "minutes",
        end="\r",
    )


print("Saving video ...")
rec_left.stop()
print("Video saved")
