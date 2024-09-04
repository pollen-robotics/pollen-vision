import time
import cv2
import depthai as dai
import numpy as np

pipeline = dai.Pipeline()

tof = pipeline.create(dai.node.ToF)

tofConfig = tof.initialConfig.get()
tofConfig.enableOpticalCorrection = False
tofConfig.enablePhaseShuffleTemporalFilter = True
tofConfig.phaseUnwrappingLevel = 4
tofConfig.phaseUnwrapErrorThreshold = 300
tofConfig.enableTemperatureCorrection = False  # Not yet supported

cam_tof = pipeline.create(dai.node.Camera)
cam_tof.setFps(30)
cam_tof.setBoardSocket(dai.CameraBoardSocket.CAM_D)
cam_tof.raw.link(tof.input)

cam_left = pipeline.create(dai.node.ColorCamera)
cam_left.setBoardSocket(dai.CameraBoardSocket.CAM_C)

undistort_manip = pipeline.createImageManip()
mesh, meshWidth, meshHeight = get_mesh(self.cam_config, cam_name)
manip.setWarpMesh(mesh, meshWidth, meshHeight)
