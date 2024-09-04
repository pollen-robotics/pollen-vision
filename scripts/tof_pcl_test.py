from pollen_vision.perception.utils.pcl_visualizer import PCLVisualizer
import time

import cv2
import depthai as dai
import numpy as np


def create_pipeline():
    pipeline = dai.Pipeline()

    tof = pipeline.create(dai.node.ToF)

    # Configure the ToF node
    tofConfig = tof.initialConfig.get()

    # Optional. Best accuracy, but adds motion blur.
    # see ToF node docs on how to reduce/eliminate motion blur.
    tofConfig.enableOpticalCorrection = False
    tofConfig.enablePhaseShuffleTemporalFilter = True
    tofConfig.phaseUnwrappingLevel = 4
    tofConfig.phaseUnwrapErrorThreshold = 300

    tofConfig.enableTemperatureCorrection = False  # Not yet supported

    xinTofConfig = pipeline.create(dai.node.XLinkIn)
    xinTofConfig.setStreamName("tofConfig")
    xinTofConfig.out.link(tof.inputConfig)

    tof.initialConfig.set(tofConfig)

    cam_tof = pipeline.create(dai.node.Camera)
    cam_tof.setFps(60)  # ToF node will produce depth frames at /2 of this rate
    cam_tof.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_tof.raw.link(tof.input)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("depth")
    tof.depth.link(xout.input)

    tofConfig = tof.initialConfig.get()

    return pipeline, tofConfig


K = np.eye(3)
K[0][0] = 100
K[1][1] = 100
K[0][2] = 320
K[1][2] = 240
P = PCLVisualizer(K)

pipeline, tofConfig = create_pipeline()
rgb = np.zeros((480, 640, 3), dtype=np.uint8)
rgb[:, :, 0] = 255
with dai.Device(pipeline) as device:
    qDepth = device.getOutputQueue(name="depth")
    while True:
        imgFrame = qDepth.get()  # blocking call, will wait until a new data has arrived
        depth_map = imgFrame.getFrame()
        depth_map = depth_map * 0.001
        print("min", min(depth_map.reshape(-1)), "max", max(depth_map.reshape(-1)))
        P.update(rgb, depth_map.astype(np.float32))
        P.tick()
        cv2.imshow("depth", depth_map)
        cv2.waitKey(1)
