import numpy as np
import cv2
import depthai as dai
from datetime import timedelta
from pollen_vision.camera_wrappers.depthai.calibration.undistort import compute_undistort_maps, get_mesh
from pollen_vision.camera_wrappers.depthai.cam_config import CamConfig
from pollen_vision.camera_wrappers.depthai.utils import (
    get_config_file_path,
    get_config_files_names,
)
import subprocess as sp

FPS = 60
pipeline = dai.Pipeline()
left = pipeline.create(dai.node.ColorCamera)
left.setFps(FPS)
left.setBoardSocket(dai.CameraBoardSocket.CAM_C)
left.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1440X1080)
left.setSensorCrop(208 / 1440, 28 / 1080)
left.setVideoSize(1024, 1024)
# left.setIspScale(2, 3)

right = pipeline.create(dai.node.ColorCamera)
right.setFps(FPS)
right.setBoardSocket(dai.CameraBoardSocket.CAM_B)
right.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1440X1080)
right.setSensorCrop(208 / 1440, 28 / 1080)
right.setVideoSize(1024, 1024)
# right.setIspScale(2, 3)

# cam_tof = pipeline.create(dai.node.Camera)
# cam_tof.setFps(30)
# cam_tof.setBoardSocket(dai.CameraBoardSocket.CAM_D)
# tof = pipeline.create(dai.node.ToF)


xout_left = pipeline.createXLinkOut()
xout_left.setStreamName("left")

xout_right = pipeline.createXLinkOut()
xout_right.setStreamName("right")

# xout_depth = pipeline.createXLinkOut()
# xout_depth.setStreamName("depth")

# tofConfig = tof.initialConfig.get()
# tofConfig.enableFPPNCorrection = True
# tofConfig.enableOpticalCorrection = False
# tofConfig.enablePhaseShuffleTemporalFilter = False
# tofConfig.phaseUnwrappingLevel = 1
# tofConfig.phaseUnwrapErrorThreshold = 300
# tofConfig.enableTemperatureCorrection = False  # Not yet supported
# tofConfig.enableWiggleCorrection = False
# tofConfig.median = dai.MedianFilter.KERNEL_3x3
# tof.initialConfig.set(tofConfig)

# Create encoders
profile = dai.VideoEncoderProperties.Profile.H264_BASELINE
bitrate = 4000
numBFrames = 0  # gstreamer recommends 0 B frames
left_encoder = pipeline.create(dai.node.VideoEncoder)
left_encoder.setDefaultProfilePreset(FPS, profile)
left_encoder.setKeyframeFrequency(FPS)  # every 1s
left_encoder.setNumBFrames(numBFrames)
left_encoder.setBitrateKbps(bitrate)

right_encoder = pipeline.create(dai.node.VideoEncoder)
right_encoder.setDefaultProfilePreset(60, profile)
right_encoder.setKeyframeFrequency(60)  # every 1s
right_encoder.setNumBFrames(numBFrames)
right_encoder.setBitrateKbps(bitrate)

# Create manip
# resolution = (960, 720)
resolution = (1024, 1024)
config_json = get_config_file_path("CONFIG_IMX296_TOF")
cam_config = CamConfig(
    cam_config_json=config_json,
    fps=FPS,
    resize=resolution,
    exposure_params=None,
    mx_id=None,
    # isp_scale=(2, 3),
    rectify=True,
)
device = dai.Device()
cam_config.set_sensor_resolution(tuple(resolution))
width_undistort_resolution = int(1024 * (cam_config.isp_scale[0] / cam_config.isp_scale[1]))
height_unistort_resolution = int(1024 * (cam_config.isp_scale[0] / cam_config.isp_scale[1]))
cam_config.set_undistort_resolution((width_undistort_resolution, height_unistort_resolution))
cam_config.set_calib(device.readCalibration())
device.close()
mapXL, mapYL, mapXR, mapYR = compute_undistort_maps(cam_config)
cam_config.set_undistort_maps(mapXL, mapYL, mapXR, mapYR)


left_manip = pipeline.createImageManip()
# try:
#     mesh, meshWidth, meshHeight = get_mesh(cam_config, "left")
#     left_manip.setWarpMesh(mesh, meshWidth, meshHeight)
# except Exception as e:
#     print(e)
#     exit()
left_manip.setMaxOutputFrameSize(resolution[0] * resolution[1] * 3)
left_manip.initialConfig.setKeepAspectRatio(True)
left_manip.initialConfig.setResize(resolution[0], resolution[1])
left_manip.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)


right_manip = pipeline.createImageManip()
# try:
#     mesh, meshWidth, meshHeight = get_mesh(cam_config, "right")
#     right_manip.setWarpMesh(mesh, meshWidth, meshHeight)
# except Exception as e:
#     print(e)
#     exit()
right_manip.setMaxOutputFrameSize(resolution[0] * resolution[1] * 3)
right_manip.initialConfig.setKeepAspectRatio(True)
right_manip.initialConfig.setResize(resolution[0], resolution[1])
right_manip.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)

# left.video.link(left_manip.inputImage)
# left_manip.out.link(left_encoder.input)
left.video.link(xout_left.input)


# right.video.link(right_manip.inputImage)
# right_manip.out.link(right_encoder.input)
right.video.out.link(xout_right.input)

# left_encoder.bitstream.link(xout_left.input)
# right_encoder.bitstream.link(xout_right.input)

# cam_tof.raw.link(tof.input)
# tof.depth.link(xout_depth.input)

device = dai.Device(pipeline)

left_queue = device.getOutputQueue(name="left", maxSize=1, blocking=False)
right_queue = device.getOutputQueue(name="right", maxSize=1, blocking=False)
# depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)


def spawn_procs(names):  # type: ignore [type-arg]
    # width, height = 1280, 720
    # width, height = 1440, 1080
    width, height = 1024, 1024
    command = [
        "ffplay",
        "-i",
        "-",
        "-x",
        str(width),
        "-y",
        str(height),
        "-framerate",
        str(FPS),
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-framedrop",
        "-strict",
        "experimental",
    ]

    procs = {}
    try:
        for name in names:
            procs[name] = sp.Popen(command, stdin=sp.PIPE)  # Start the ffplay process
    except Exception:
        exit("Error: cannot run ffplay!\nTry running: sudo apt install ffmpeg")

    return procs


latencies_left = []
latencies_right = []
# procs = spawn_procs(["left", "right"])
while True:
    latency = {}

    left_frame = left_queue.get()
    latency["left"] = dai.Clock.now() - left_frame.getTimestamp()
    right_frame = right_queue.get()
    latency["right"] = dai.Clock.now() - right_frame.getTimestamp()
    # depth_frame = depth_queue.get()
    # latency["depth"] = dai.Clock.now() - depth_frame.getTimestamp()

    # io = procs["left"].stdin
    # io.write(left_frame.getData())

    # io = procs["right"].stdin
    # io.write(right_frame.getData())

    cv2.imshow("left", left_frame.getCvFrame())
    cv2.imshow("right", right_frame.getCvFrame())
    # cv2.imshow("depth", depth_frame.getFrame())
    # print(latency)
    latencies_left.append(latency["left"])
    latencies_right.append(latency["right"])

    print(np.mean(latencies_left[-50:]), np.mean(latencies_right[-50:]))

    if cv2.waitKey(1) == ord("q"):
        break
