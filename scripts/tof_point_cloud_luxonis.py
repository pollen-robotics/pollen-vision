import datetime
import os
import sys
from datetime import timedelta

import cv2
import depthai as dai
import numpy as np
from pollen_vision.camera_wrappers.depthai.utils import colorizeDepth

try:
    import open3d as o3d
except ImportError:
    sys.exit("Open3D missing. Install it using the command: '{} -m pip install open3d'".format(sys.executable))

FPS = 30

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
pointcloud = pipeline.create(dai.node.PointCloud)

# ToF settings
camTof.setFps(FPS)
camTof.setBoardSocket(TOF_SOCKET)

tofConfig = tof.initialConfig.get()
# tofConfig.enableOpticalCorrection = False
tofConfig.phaseUnwrappingLevel = 4
# choose a median filter or use none - using the median filter improves the pointcloud but causes discretization of the data
# tofConfig.median = dai.MedianFilter.MEDIAN_OFF
tofConfig.median = dai.MedianFilter.KERNEL_3x3
# tofConfig.median = dai.MedianFilter.KERNEL_5x5
# tofConfig.median = dai.MedianFilter.KERNEL_7x7
# tofConfig.enableDistortionCorrection = False
tof.initialConfig.set(tofConfig)

# rgb settings
camRgb.setBoardSocket(RGB_SOCKET)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
camRgb.setFps(FPS)
camRgb.setIspScale(1, 2)

out.setStreamName("out")

sync.setSyncThreshold(timedelta(seconds=(1 / FPS)))

# Linking
camRgb.isp.link(sync.inputs["rgb"])
camTof.raw.link(tof.input)
tof.depth.link(align.input)
align.outputAligned.link(sync.inputs["depth_aligned"])
align.outputAligned.link(pointcloud.inputDepth)
sync.inputs["rgb"].setBlocking(False)
camRgb.isp.link(align.inputAlignTo)
pointcloud.outputPointCloud.link(sync.inputs["pcl"])
sync.out.link(out.input)
out.setStreamName("out")

with dai.Device(pipeline) as device:
    isRunning = True

    q = device.getOutputQueue(name="out", maxSize=4, blocking=False)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()
    coordinateFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000, origin=[0, 0, 0])
    # vis.add_geometry(coordinateFrame)

    first = True

    # Create a ViewControl object
    view_control = vis.get_view_control()

    while isRunning:
        inMessage = q.get()
        inColor = inMessage["rgb"]  # type: ignore
        inPointCloud = inMessage["pcl"]  # type: ignore
        inDepth = inMessage["depth_aligned"]  # type: ignore

        cvColorFrame = inColor.getCvFrame()
        # Convert the frame to RGB
        cvRGBFrame = cv2.cvtColor(cvColorFrame, cv2.COLOR_BGR2RGB)
        cv2.imshow("color", cvColorFrame)
        colorized_depth = colorizeDepth(inDepth.getCvFrame())
        cv2.imshow("depth", colorized_depth)
        blended = cv2.addWeighted(cvColorFrame, 0.5, colorized_depth, 0.5, 0)
        cv2.imshow("blended", blended)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        if key == ord("c"):
            print("saving...")
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
            new_output = formatted_time
            os.mkdir(new_output)
            o3d.io.write_point_cloud(os.path.join(new_output, "tof_pointcloud.ply"), pcd)
            cv2.imwrite(os.path.join(new_output, "rgb.png"), cvColorFrame)
            print(f"RGB point cloud saved to folder {new_output}")
        if inPointCloud:
            points = inPointCloud.getPoints().astype(np.float64)
            points[:, 0] = -points[:, 0]  # Invert X axis
            # points[:, 1] = -points[:, 1]  # Invert Y axis
            # points[:, 2] = points[:, 2] / 1000.0  # Convert Z axis from mm to meters (if needed)

            pcd.points = o3d.utility.Vector3dVector(points)
            colors = (cvRGBFrame.reshape(-1, 3) / 255.0).astype(np.float64)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            if first:
                vis.add_geometry(pcd)
                first = False
            else:
                vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
    vis.destroy_window()