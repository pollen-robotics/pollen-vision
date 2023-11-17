import json
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import cv2
import depthai as dai
import numpy as np

from depthai_wrappers.cam_config import CamConfig
from depthai_wrappers.utils import get_inv_R_T, get_socket_from_name


class Wrapper:
    def __init__(
        self,
        cam_config_json: str,
        fps: int,
        force_usb2: bool,
        resize: Tuple[int, int],
        rectify: bool,
        exposure_params: Tuple[int, int],
    ) -> None:
        self.cam_config = CamConfig(cam_config_json, fps, resize, exposure_params)
        self.force_usb2 = force_usb2
        self.rectify = rectify

        self.prepare()

    def prepare(self) -> None:
        connected_cameras_features = dai.Device().getConnectedCameraFeatures()

        # Assuming both cameras are the same
        width = connected_cameras_features[0].width
        height = connected_cameras_features[0].height

        self.cam_config.set_sensor_resolution((width, height))
        self.cam_config.set_undistort_resolution(
            (960, 720)
        )  # TODO find a way to get this from cam.ispsize()
        self.compute_undistort_maps()

        self.pipeline = self.create_pipeline()

        self.device = dai.Device(
            self.pipeline,
            maxUsbSpeed=(
                dai.UsbSpeed.HIGH if self.force_usb2 else dai.UsbSpeed.SUPER_PLUS
            ),
        )
        self.queues = self.create_queues()

        self.print_info()

    def print_info(self) -> None:
        print("==================")
        print(self.cam_config.to_string())
        print("Force USB2: {}".format(self.force_usb2))
        print("Rectify: {}".format(self.rectify))
        print("==================")

    def get_data(self) -> tuple:
        data: dict = {}
        latency: dict = {}
        ts: dict = {}

        for name, queue in self.queues.items():
            pkt = queue.get()
            data[name] = pkt
            latency[name] = dai.Clock.now() - pkt.getTimestamp()
            ts[name] = pkt.getTimestamp()

        return data, latency, ts

    def create_pipeline(self) -> dai.Pipeline:
        print("Abstract class Wrapper does not implement create_pipeline()")
        exit()

    def pipeline_basis(self) -> dai.Pipeline:
        pipeline = dai.Pipeline()

        left_socket = get_socket_from_name("left", self.cam_config.name_to_socket)
        right_socket = get_socket_from_name("right", self.cam_config.name_to_socket)

        self.left = pipeline.createColorCamera()
        self.left.setFps(self.cam_config.fps)
        self.left.setBoardSocket(left_socket)
        self.left.setResolution(
            dai.ColorCameraProperties.SensorResolution.THE_1440X1080
        )
        self.left.setIspScale(2, 3)  # -> 960, 720

        self.right = pipeline.createColorCamera()
        self.right.setFps(self.cam_config.fps)
        self.right.setBoardSocket(right_socket)
        self.right.setResolution(
            dai.ColorCameraProperties.SensorResolution.THE_1440X1080
        )
        self.right.setIspScale(2, 3)  # -> 960, 720

        # self.cam_config.set_undistort_resolution(self.left.getIspSize())

        if self.cam_config.exposure_params is not None:
            self.left.initialControl.setManualExposure(*self.cam_config.exposure_params)
            self.right.initialControl.setManualExposure(
                *self.cam_config.exposure_params
            )
        if self.cam_config.inverted:
            self.left.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
            self.right.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)

        self.left_manipRectify = self.create_manipRectify(
            pipeline, "left", self.cam_config.undistort_resolution, self.rectify
        )
        self.right_manipRectify = self.create_manipRectify(
            pipeline, "right", self.cam_config.undistort_resolution, self.rectify
        )

        self.left_manipRescale = self.create_manipResize(
            pipeline, self.cam_config.resize_resolution
        )
        self.right_manipRescale = self.create_manipResize(
            pipeline, self.cam_config.resize_resolution
        )
        return pipeline

    def link_pipeline(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        print("Abstract class Wrapper does not implement link_pipeline()")
        exit()

    def create_output_streams(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        self.xout_left = pipeline.createXLinkOut()
        self.xout_left.setStreamName("left")

        self.xout_right = pipeline.createXLinkOut()
        self.xout_right.setStreamName("right")

        return pipeline

    def create_queues(self) -> dict[str, dai.DataOutputQueue]:
        queues = {}
        for name in ["left", "right"]:
            queues[name] = self.device.getOutputQueue(name, maxSize=1, blocking=False)
        return queues

    def create_manipRectify(
        self,
        pipeline: dai.Pipeline,
        cam_name: str,
        resolution: Tuple[int, int],
        rectify: bool = True,
    ) -> dai.node.ImageManip:
        manipRectify = pipeline.createImageManip()

        if rectify:
            try:
                mesh, meshWidth, meshHeight = self.get_mesh(cam_name)
                manipRectify.setWarpMesh(mesh, meshWidth, meshHeight)
            except Exception as e:
                print(e)
                exit()

        manipRectify.setMaxOutputFrameSize(resolution[0] * resolution[1] * 3)
        return manipRectify

    def create_manipResize(
        self, pipeline: dai.Pipeline, resolution: Tuple[int, int]
    ) -> dai.node.ImageManip:
        manipResize = pipeline.createImageManip()
        manipResize.initialConfig.setResizeThumbnail(resolution[0], resolution[1])
        manipResize.setMaxOutputFrameSize(resolution[0] * resolution[1] * 3)

        # TODO add this in child method for teleop
        # manipResize.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)

        return manipResize

    def compute_undistort_maps(self) -> None:
        left_socket = get_socket_from_name("left", self.cam_config.name_to_socket)
        right_socket = get_socket_from_name("right", self.cam_config.name_to_socket)

        resolution = self.cam_config.undistort_resolution

        calib = dai.Device().readCalibration()
        left_K = np.array(
            calib.getCameraIntrinsics(
                left_socket,
                resolution[0],
                resolution[1],
            )
        )
        left_D = np.array(calib.getDistortionCoefficients(left_socket))

        right_K = np.array(
            calib.getCameraIntrinsics(
                right_socket,
                resolution[0],
                resolution[1],
            )
        )
        right_D = np.array(calib.getDistortionCoefficients(right_socket))
        R = np.array(calib.getStereoRightRectificationRotation())

        T = np.array(calib.getCameraTranslationVector(left_socket, right_socket))
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            left_K,
            left_D,
            right_K,
            right_D,
            resolution,
            R,
            T,
            flags=0,
        )

        if self.cam_config.fisheye:
            mapXL, mapYL = cv2.fisheye.initUndistortRectifyMap(
                left_K, left_D, R1, P1, resolution, cv2.CV_32FC1
            )
            mapXR, mapYR = cv2.fisheye.initUndistortRectifyMap(
                right_K, right_D, R2, P2, resolution, cv2.CV_32FC1
            )
        else:
            mapXL, mapYL = cv2.initUndistortRectifyMap(
                left_K, left_D, R1, P1, resolution, cv2.CV_32FC1
            )
            mapXR, mapYR = cv2.initUndistortRectifyMap(
                right_K, right_D, R2, P2, resolution, cv2.CV_32FC1
            )

        self.cam_config.set_undistort_maps(mapXL, mapYL, mapXR, mapYR)

    def get_mesh(self, cam_name: str) -> Tuple[np.ndarray, int, int]:
        mapX, mapY = self.cam_config.undstort_maps[cam_name]
        if mapX is None or mapY is None:
            raise Exception(
                "Undistort maps have not been computed. Call compute_undistort_maps() first."
            )

        meshCellSize = 16
        mesh0 = []
        for y in range(mapX.shape[0] + 1):
            if y % meshCellSize == 0:
                rowLeft = []
                for x in range(mapX.shape[1]):
                    if x % meshCellSize == 0:
                        if y == mapX.shape[0] and x == mapX.shape[1]:
                            rowLeft.append(mapX[y - 1, x - 1])
                            rowLeft.append(mapY[y - 1, x - 1])
                        elif y == mapX.shape[0]:
                            rowLeft.append(mapX[y - 1, x])
                            rowLeft.append(mapY[y - 1, x])
                        elif x == mapX.shape[1]:
                            rowLeft.append(mapX[y, x - 1])
                            rowLeft.append(mapY[y, x - 1])
                        else:
                            rowLeft.append(mapX[y, x])
                            rowLeft.append(mapY[y, x])
                if (mapX.shape[1] % meshCellSize) % 2 != 0:
                    rowLeft.append(0)
                    rowLeft.append(0)

                mesh0.append(rowLeft)

        mesh0 = np.array(mesh0)
        meshWidth = mesh0.shape[1] // 2
        meshHeight = mesh0.shape[0]
        mesh0.resize(meshWidth * meshHeight, 2)

        mesh = list(map(tuple, mesh0))

        return mesh, meshWidth, meshHeight

    # Takes in the output of multical calibration
    def flash(self, calib_json_file: str) -> bool:
        now = str(datetime.now()).replace(" ", "_").split(".")[0]

        device_calibration_backup_file = Path("./CALIBRATION_BACKUP_" + now + ".json")
        deviceCalib = self.device.readCalibration()
        deviceCalib.eepromToJsonFile(device_calibration_backup_file)
        print("Backup of device calibration saved to", device_calibration_backup_file)

        os.environ["DEPTHAI_ALLOW_FACTORY_FLASHING"] = "235539980"

        ch = dai.CalibrationHandler()
        calibration_data = json.load(open(calib_json_file, "rb"))

        cameras = calibration_data["cameras"]
        camera_poses = calibration_data["camera_poses"]

        print("Setting intrinsics ...")
        for cam_name, params in cameras.items():
            K = np.array(params["K"])
            D = np.array(params["dist"]).reshape((-1))
            im_size = params["image_size"]
            cam_socket = get_socket_from_name(cam_name, self.cam_config.name_to_socket)

            ch.setCameraIntrinsics(cam_socket, K, im_size)
            ch.setDistortionCoefficients(cam_socket, D)
            if self.cam_config.fisheye:
                print("Setting camera type to fisheye ...")
                ch.setCameraType(cam_socket, dai.CameraModel.Fisheye)

        print("Setting extrinsics ...")
        left_socket = get_socket_from_name("left", self.cam_config.name_to_socket)
        right_socket = get_socket_from_name("right", self.cam_config.name_to_socket)

        right_to_left = camera_poses["right_to_left"]
        R_right_to_left = np.array(right_to_left["R"])
        T_right_to_left = np.array(right_to_left["T"])
        T_right_to_left *= 100  # Needs to be in centimeters (?) # TODO test

        R_left_to_right, T_left_to_right = get_inv_R_T(R_right_to_left, T_right_to_left)

        ch.setCameraExtrinsics(
            left_socket,
            right_socket,
            R_right_to_left,
            T_right_to_left,
            specTranslation=T_right_to_left,
        )
        ch.setCameraExtrinsics(
            right_socket,
            left_socket,
            R_left_to_right,
            T_left_to_right,
            specTranslation=T_left_to_right,
        )

        ch.setStereoLeft(left_socket, np.eye(3))
        ch.setStereoRight(right_socket, R_right_to_left)

        print("Flashing ...")
        try:
            self.device.flashCalibration2(ch)
            print("Calibration flashed successfully")
            return True
        except Exception as e:
            print("Flashing failed")
            print(e)
            return False

    def close(self) -> None:
        self.device.close()
