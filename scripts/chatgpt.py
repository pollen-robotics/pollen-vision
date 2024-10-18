import depthai as dai
import cv2
import numpy as np
import time


# Define a function to colorize the depth map
def colorizeDepth(frameDepth):
    invalidMask = frameDepth == 0
    try:
        minDepth = np.percentile(frameDepth[frameDepth != 0], 3)
        maxDepth = np.percentile(frameDepth[frameDepth != 0], 95)
        logDepth = np.log(frameDepth, where=frameDepth != 0)
        logMinDepth = np.log(minDepth)
        logMaxDepth = np.log(maxDepth)
        np.nan_to_num(logDepth, copy=False, nan=logMinDepth)
        logDepth = np.clip(logDepth, logMinDepth, logMaxDepth)
        depthFrameColor = np.interp(logDepth, (logMinDepth, logMaxDepth), (0, 255))
        depthFrameColor = np.nan_to_num(depthFrameColor).astype(np.uint8)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
        depthFrameColor[invalidMask] = 0
    except IndexError:
        depthFrameColor = np.zeros((frameDepth.shape[0], frameDepth.shape[1], 3), dtype=np.uint8)
    except Exception as e:
        raise e
    return depthFrameColor


# Create a pipeline
pipeline = dai.Pipeline()

# Create RGB and ToF camera nodes
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_C)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
camRgb.setFps(30)

camTof = pipeline.create(dai.node.Camera)
camTof.setFps(30)
camTof.setBoardSocket(dai.CameraBoardSocket.CAM_D)

# Create ToF node
tof = pipeline.create(dai.node.ToF)
camTof.raw.link(tof.input)

# Create align node to align ToF depth to RGB
align = pipeline.create(dai.node.ImageAlign)
tof.depth.link(align.input)
camRgb.isp.link(align.inputAlignTo)

# Create outputs
rgbOut = pipeline.create(dai.node.XLinkOut)
rgbOut.setStreamName("rgb")
alignOut = pipeline.create(dai.node.XLinkOut)
alignOut.setStreamName("depth_aligned")

camRgb.isp.link(rgbOut.input)
align.outputAligned.link(alignOut.input)

# Function to update blend weights for depth and RGB overlay
rgbWeight = 0.4
depthWeight = 0.6


def updateBlendWeights(percentRgb):
    global depthWeight, rgbWeight
    rgbWeight = float(percentRgb) / 100.0
    depthWeight = 1.0 - rgbWeight


# Start the pipeline
with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue("rgb", 8, False)
    qDepthAligned = device.getOutputQueue("depth_aligned", 8, False)

    cv2.namedWindow("RGB-Depth")
    cv2.createTrackbar("RGB Weight %", "RGB-Depth", int(rgbWeight * 100), 100, updateBlendWeights)

    while True:
        inRgb = qRgb.get()
        inDepthAligned = qDepthAligned.get()

        frameRgb = inRgb.getCvFrame()
        frameDepthAligned = inDepthAligned.getCvFrame()

        depthColorized = colorizeDepth(frameDepthAligned)

        # Blend the RGB image with the colorized depth image
        blended = cv2.addWeighted(frameRgb, rgbWeight, depthColorized, depthWeight, 0)
        cv2.imshow("RGB-Depth", blended)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
