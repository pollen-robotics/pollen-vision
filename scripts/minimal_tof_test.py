#!/usr/bin/env python3

import time
import cv2
import depthai as dai
import numpy as np

print(dai.__version__)

cvColorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
cvColorMap[0] = [0, 0, 0]


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
    cam_tof.setFps(30)  # ToF node will produce depth frames at /2 of this rate
    cam_tof.setBoardSocket(dai.CameraBoardSocket.CAM_D)
    cam_tof.raw.link(tof.input)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("depth")
    tof.depth.link(xout.input)

    tofConfig = tof.initialConfig.get()

    left = pipeline.create(dai.node.ColorCamera)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    left.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1440X1080)
    left.setFps(30)
    left.setIspScale(2, 3)

    left_xout = pipeline.create(dai.node.XLinkOut)
    left_xout.setStreamName("left")
    left.video.link(left_xout.input)

    return pipeline, tofConfig


if __name__ == "__main__":
    pipeline, tofConfig = create_pipeline()

    with dai.Device(pipeline) as device:
        print("Connected cameras:", device.getConnectedCameraFeatures())

        # qDepth = device.getOutputQueue(name="depth")
        qDepth = device.getOutputQueue(name="depth", maxSize=8, blocking=False)

        # left_q = device.getOutputQueue(name="left")
        left_q = device.getOutputQueue(name="left", maxSize=8, blocking=False)

        while True:
            start = time.time()

            imgFrame = qDepth.get()  # blocking call, will wait until a new data has arrived
            depth_map = imgFrame.getFrame()

            max_depth = (tofConfig.phaseUnwrappingLevel + 1) * 1500  # 100MHz modulation freq.
            depth_raw = depth_map / max_depth
            depth_colorized = np.interp(depth_map, (0, max_depth), (0, 255)).astype(np.uint8)
            depth_colorized = cv2.applyColorMap(depth_colorized, cvColorMap)

            # If I comment that (3 next lines), the tof works fine
            in_left = left_q.get()
            left_im = in_left.getCvFrame()
            cv2.imshow("left", left_im)
            # Until that

            cv2.imshow("Colorized depth", depth_colorized)
            cv2.imshow("depth raw", depth_raw)
            key = cv2.waitKey(1)

    device.close()
