import depthai as dai
import cv2

print(dai.__version__)

pipeline = dai.Pipeline()

cam_tof = pipeline.create(dai.node.Camera)
cam_tof.setFps(30)

cam_tof.setBoardSocket(dai.CameraBoardSocket.CAM_D)

tof = pipeline.create(dai.node.ToF)

tofConfig = tof.initialConfig.get()
tofConfig.enableDistortionCorrection = False
tof.initialConfig.set(tofConfig)
cam_tof.raw.link(tof.input)

output = pipeline.create(dai.node.XLinkOut)
output.setStreamName("intensity")

tof.intensity.link(output.input)

device = dai.Device(pipeline)

q = device.getOutputQueue(name="intensity", maxSize=8, blocking=False)

while True:
    data = q.get()
    cv2.imshow("intensity", data.getCvFrame())
    cv2.waitKey(1)
