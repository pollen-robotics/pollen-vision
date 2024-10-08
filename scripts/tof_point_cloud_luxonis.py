import depthai as dai
import numpy as np
import cv2
import time
from datetime import timedelta
import datetime
import os
import sys

try:
    import open3d as o3d
except ImportError:
    sys.exit(
        "Critical dependency missing: Open3D. Please install it using the command: '{} -m pip install open3d' and then rerun the script.".format(
            sys.executable
        )
    )

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


def colorizeDepth(frameDepth):
    invalidMask = frameDepth == 0
    # Log the depth, minDepth and maxDepth
    try:
        minDepth = np.percentile(frameDepth[frameDepth != 0], 3)
        maxDepth = np.percentile(frameDepth[frameDepth != 0], 95)
        logDepth = np.log(frameDepth, where=frameDepth != 0)
        logMinDepth = np.log(minDepth)
        logMaxDepth = np.log(maxDepth)
        np.nan_to_num(logDepth, copy=False, nan=logMinDepth)
        # Clip the values to be in the 0-255 range
        logDepth = np.clip(logDepth, logMinDepth, logMaxDepth)

        # Interpolate only valid logDepth values, setting the rest based on the mask
        depthFrameColor = np.interp(logDepth, (logMinDepth, logMaxDepth), (0, 255))
        depthFrameColor = np.nan_to_num(depthFrameColor)
        depthFrameColor = depthFrameColor.astype(np.uint8)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
        # Set invalid depth pixels to black
        depthFrameColor[invalidMask] = 0
    except IndexError:
        # Frame is likely empty
        depthFrameColor = np.zeros((frameDepth.shape[0], frameDepth.shape[1], 3), dtype=np.uint8)
    except Exception as e:
        raise e
    return depthFrameColor


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
        inColor = inMessage["rgb"]
        inPointCloud = inMessage["pcl"]
        inDepth = inMessage["depth_aligned"]

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
