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
    # print(dir(tof.properties.initialConfig))
    # print(tof.properties.initialConfig.data)
    # exit()
    # Configure the ToF node
    tofConfig = tof.initialConfig.get()

    # Optional. Best accuracy, but adds motion blur.
    # see ToF node docs on how to reduce/eliminate motion blur.
    tofConfig.enableFPPNCorrection = True
    tofConfig.enableOpticalCorrection = False
    tofConfig.enablePhaseShuffleTemporalFilter = False
    tofConfig.phaseUnwrappingLevel = 1
    tofConfig.phaseUnwrapErrorThreshold = 300
    tofConfig.enableTemperatureCorrection = False  # Not yet supported
    tofConfig.enableWiggleCorrection = False
    tofConfig.median = dai.MedianFilter.KERNEL_3x3

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

    right = pipeline.create(dai.node.ColorCamera)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    right.setIspScale(2, 3)
    left = pipeline.create(dai.node.ColorCamera)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    left.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1440X1080)
    left.setFps(30)
    left.setIspScale(2, 3)

    left_xout = pipeline.create(dai.node.XLinkOut)
    left_xout.setStreamName("left")
    left.video.link(left_xout.input)

    right_xout = pipeline.create(dai.node.XLinkOut)
    right_xout.setStreamName("right")
    right.video.link(right_xout.input)

    return pipeline, tofConfig


mouse_x, mouse_y = 0, 0


# mouse callback function
def cb(event, x, y, flags, param):
    global mouse_x, mouse_y
    mouse_x = x
    mouse_y = y


if __name__ == "__main__":
    # global mouse_x, mouse_y
    pipeline, tofConfig = create_pipeline()
    cv2.namedWindow("Colorized depth")
    cv2.setMouseCallback("Colorized depth", cb)

    with dai.Device(pipeline) as device:
        print("Connected cameras:", device.getConnectedCameraFeatures())
        qDepth = device.getOutputQueue(name="depth", maxSize=8, blocking=False)

        left_q = device.getOutputQueue(name="left", maxSize=8, blocking=False)
        right_q = device.getOutputQueue(name="right", maxSize=8, blocking=False)

        tofConfigInQueue = device.getInputQueue("tofConfig")

        counter = 0
        while True:
            start = time.time()
            key = cv2.waitKey(1)
            if key == ord("f"):
                tofConfig.enableFPPNCorrection = not tofConfig.enableFPPNCorrection
                tofConfigInQueue.send(tofConfig)
            elif key == ord("o"):
                tofConfig.enableOpticalCorrection = not tofConfig.enableOpticalCorrection
                tofConfigInQueue.send(tofConfig)
            elif key == ord("w"):
                tofConfig.enableWiggleCorrection = not tofConfig.enableWiggleCorrection
                tofConfigInQueue.send(tofConfig)
            elif key == ord("t"):
                tofConfig.enableTemperatureCorrection = not tofConfig.enableTemperatureCorrection
                tofConfigInQueue.send(tofConfig)
            elif key == ord("q"):
                break
            elif key == ord("0"):
                tofConfig.enablePhaseUnwrapping = False
                tofConfig.phaseUnwrappingLevel = 0
                tofConfigInQueue.send(tofConfig)
            elif key == ord("1"):
                tofConfig.enablePhaseUnwrapping = True
                tofConfig.phaseUnwrappingLevel = 1
                tofConfigInQueue.send(tofConfig)
            elif key == ord("2"):
                tofConfig.enablePhaseUnwrapping = True
                tofConfig.phaseUnwrappingLevel = 2
                tofConfigInQueue.send(tofConfig)
            elif key == ord("3"):
                tofConfig.enablePhaseUnwrapping = True
                tofConfig.phaseUnwrappingLevel = 3
                tofConfigInQueue.send(tofConfig)
            elif key == ord("4"):
                tofConfig.enablePhaseUnwrapping = True
                tofConfig.phaseUnwrappingLevel = 4
                tofConfigInQueue.send(tofConfig)
            elif key == ord("5"):
                tofConfig.enablePhaseUnwrapping = True
                tofConfig.phaseUnwrappingLevel = 5
                tofConfigInQueue.send(tofConfig)
            elif key == ord("m"):
                medianSettings = [
                    dai.MedianFilter.MEDIAN_OFF,
                    dai.MedianFilter.KERNEL_3x3,
                    dai.MedianFilter.KERNEL_5x5,
                    dai.MedianFilter.KERNEL_7x7,
                ]
                currentMedian = tofConfig.median
                nextMedian = medianSettings[(medianSettings.index(currentMedian) + 1) % len(medianSettings)]
                print(f"Changing median to {nextMedian.name} from {currentMedian.name}")
                tofConfig.median = nextMedian
                tofConfigInQueue.send(tofConfig)

            print("===")
            print("enableFPPNCorrection", tofConfig.enableFPPNCorrection)
            print("enableOpticalCorrection", tofConfig.enableOpticalCorrection)
            print("enableWiggleCorrection", tofConfig.enableWiggleCorrection)
            print("enableTemperatureCorrection", tofConfig.enableTemperatureCorrection)
            print("enablePhaseUnwrapping", tofConfig.enablePhaseUnwrapping)
            print("phaseUnwrappingLevel", tofConfig.phaseUnwrappingLevel)
            print("median", tofConfig.median)
            print("===")
            imgFrame = qDepth.get()  # blocking call, will wait until a new data has arrived
            depth_map = imgFrame.getFrame()

            max_depth = (tofConfig.phaseUnwrappingLevel + 1) * 1500  # 100MHz modulation freq.
            depth_colorized = np.interp(depth_map, (0, max_depth), (0, 255)).astype(np.uint8)
            depth_colorized = cv2.applyColorMap(depth_colorized, cvColorMap)

            depth_at_mouse = depth_map[mouse_y, mouse_x]
            print(mouse_x, mouse_y, depth_at_mouse)
            depth_colorized = cv2.circle(depth_colorized, (mouse_x, mouse_y), 5, (255, 255, 255), -1)

            in_left = left_q.get()
            in_right = right_q.get()

            left_im = in_left.getCvFrame()
            right_im = in_right.getCvFrame()

            cv2.imshow("left", left_im)
            cv2.imshow("right", right_im)
            cv2.imshow("Colorized depth", depth_colorized)
            counter += 1
            time.sleep(1 / 30)

    device.close()
