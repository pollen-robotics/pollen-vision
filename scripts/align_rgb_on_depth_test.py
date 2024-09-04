import numpy as np
import cv2
import depthai as dai
from datetime import timedelta

FPS = 30

pipeline = dai.Pipeline()
left = pipeline.create(dai.node.ColorCamera)
cam_tof = pipeline.create(dai.node.Camera)
tof = pipeline.create(dai.node.ToF)
sync = pipeline.create(dai.node.Sync)
align = pipeline.create(dai.node.ImageAlign)
out = pipeline.create(dai.node.XLinkOut)
out.setStreamName("out")

tofConfig = tof.initialConfig.get()
tofConfig.enableFPPNCorrection = True
tofConfig.enableOpticalCorrection = False
tofConfig.enablePhaseShuffleTemporalFilter = False
tofConfig.phaseUnwrappingLevel = 1
tofConfig.phaseUnwrapErrorThreshold = 300
tofConfig.enableTemperatureCorrection = False  # Not yet supported
tofConfig.enableWiggleCorrection = False
tofConfig.median = dai.MedianFilter.KERNEL_3x3
tof.initialConfig.set(tofConfig)

cam_tof.setFps(FPS)
cam_tof.setBoardSocket(dai.CameraBoardSocket.CAM_D)

left.setBoardSocket(dai.CameraBoardSocket.CAM_C)
left.setFps(FPS)
left.setIspScale(2, 3)

sync.setSyncThreshold(timedelta(seconds=0.5 / FPS))

# left.isp.link(sync.inputs["left"])
tof.depth.link(sync.inputs["depth"])
cam_tof.raw.link(tof.input)
tof.depth.link(align.inputAlignTo)
left.isp.link(align.input)
sync.inputs["left"].setBlocking(False)
align.outputAligned.link(sync.inputs["left_aligned"])
sync.out.link(out.input)

device = dai.Device(pipeline)

queue = device.getOutputQueue(name="out", maxSize=8, blocking=False)

while True:
    messageGroup = queue.get()
    frameDepth = messageGroup["depth"]
    frameLeft = messageGroup["left_aligned"]

    left_aligned = frameLeft.getCvFrame()
    depth = frameDepth.getFrame()

    cv2.imshow("left_aligned", left_aligned)
    cv2.imshow("depth", depth)
    cv2.waitKey(1)
