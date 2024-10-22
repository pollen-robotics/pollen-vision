from datetime import timedelta

import cv2
import depthai as dai
import numpy as np
from pollen_vision.camera_wrappers.depthai.utils import colorizeDepth

# This example is intended to run unchanged on an OAK-D-SR-PoE camera
FPS = 30.0

RGB_SOCKET = dai.CameraBoardSocket.CAM_C
TOF_SOCKET = dai.CameraBoardSocket.CAM_D
ALIGN_SOCKET = RGB_SOCKET


pipeline = dai.Pipeline()
# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
tof = pipeline.create(dai.node.ToF)
camTof = pipeline.create(dai.node.Camera)
sync = pipeline.create(dai.node.Sync)
align = pipeline.create(dai.node.ImageAlign)
out = pipeline.create(dai.node.XLinkOut)

# ToF settings
camTof.setFps(FPS)
# camTof.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
camTof.setBoardSocket(TOF_SOCKET)

# rgb settings
camRgb.setBoardSocket(RGB_SOCKET)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
camRgb.setFps(FPS)

camRgb.setIspScale(1, 2)

# depthSize = (1280, 800)  # PLEASE SET TO BE SIZE OF THE TOF STREAM
depthSize = (640, 480)  # PLEASE SET TO BE SIZE OF THE TOF STREAM
rgbSize = camRgb.getIspSize()

out.setStreamName("out")

sync.setSyncThreshold(timedelta(seconds=0.5 / FPS))
rgbSize = camRgb.getIspSize()

# Linking
camRgb.isp.link(sync.inputs["rgb"])
camTof.raw.link(tof.input)
tof.depth.link(align.input)
align.outputAligned.link(sync.inputs["depth_aligned"])
sync.inputs["rgb"].setBlocking(False)
camRgb.isp.link(align.inputAlignTo)
sync.out.link(out.input)


rgbWeight = 0.4
depthWeight = 0.6


def updateBlendWeights(percentRgb: float) -> None:
    """
    Update the rgb and depth weights used to blend depth/rgb image

    @param[in] percent_rgb The rgb weight expressed as a percentage (0..100)
    """
    global depthWeight
    global rgbWeight
    rgbWeight = float(percentRgb) / 100.0
    depthWeight = 1.0 - rgbWeight


# Connect to device and start pipeline
remapping = True
with dai.Device(pipeline) as device:
    queue = device.getOutputQueue("out", 8, False)

    # Configure windows; trackbar adjusts blending ratio of rgb/depth
    rgbDepthWindowName = "rgb-depth"

    cv2.namedWindow(rgbDepthWindowName)
    cv2.createTrackbar(
        "RGB Weight %",
        rgbDepthWindowName,
        int(rgbWeight * 100),
        100,
        updateBlendWeights,
    )
    try:
        calibData = device.readCalibration2()
        M1 = np.array(calibData.getCameraIntrinsics(ALIGN_SOCKET, *depthSize))
        D1 = np.array(calibData.getDistortionCoefficients(ALIGN_SOCKET))
        M2 = np.array(calibData.getCameraIntrinsics(RGB_SOCKET, *rgbSize))
        D2 = np.array(calibData.getDistortionCoefficients(RGB_SOCKET))

        R = np.array(calibData.getCameraExtrinsics(ALIGN_SOCKET, TOF_SOCKET, False))[0:3, 0:3]

        TARGET_MATRIX = M1

        lensPosition = calibData.getLensPosition(RGB_SOCKET)
    except Exception:
        raise
    while True:
        messageGroup = queue.get()
        assert isinstance(messageGroup, dai.MessageGroup)
        frameRgb = messageGroup["rgb"]
        assert isinstance(frameRgb, dai.ImgFrame)
        frameDepth = messageGroup["depth_aligned"]
        assert isinstance(frameDepth, dai.ImgFrame)

        sizeRgb = frameRgb.getData().size
        sizeDepth = frameDepth.getData().size
        # Blend when both received
        if frameDepth is not None:
            rgbFrame = frameRgb.getCvFrame()
            # Colorize the aligned depth
            alignedDepthColorized = colorizeDepth(frameDepth.getFrame())
            # Resize depth to match the rgb frame

            cv2.imshow("depth", alignedDepthColorized)
            key = cv2.waitKey(1)
            if key == ord("m"):
                if remapping:
                    print("Remap turned OFF.")
                    remapping = False
                else:
                    print("Remap turned ON.")
                    remapping = True

            if remapping:
                # mapX, mapY = cv2.fisheye.initUndistortRectifyMap(M2, D2, None, M2, rgbSize, cv2.CV_32FC1)
                mapX, mapY = cv2.fisheye.initUndistortRectifyMap(M2, D2, R, M2, rgbSize, cv2.CV_32FC1)
                rgbFrame = cv2.remap(rgbFrame, mapX, mapY, cv2.INTER_LINEAR)
            print(rgbFrame.shape)
            alignedDepthColorized = cv2.resize(alignedDepthColorized, (rgbFrame.shape[1], rgbFrame.shape[0]))
            blended = cv2.addWeighted(rgbFrame, rgbWeight, alignedDepthColorized, depthWeight, 0)
            cv2.imshow(rgbDepthWindowName, blended)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
