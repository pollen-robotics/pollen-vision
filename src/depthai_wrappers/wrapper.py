import json
import os

import cv2
import depthai as dai
import numpy as np

from depthai_wrappers.cam_config import CamConfig
from depthai_wrappers.utils import get_inv_R_T, get_socket_from_name


class Wrapper:
    def __init__(self, cam_config_json, fps, force_usb2, resize, rectify):
        self.cam_config = CamConfig(cam_config_json, fps, resize)
        self.force_usb2 = force_usb2
        self.rectify = rectify

        self.prepare()

    def prepare(self):
        connected_cameras_features = dai.Device().getConnectedCameraFeatures()

        # Assuming both cameras are the same
        width = connected_cameras_features[0].width
        height = connected_cameras_features[0].height

        self.cam_config.set_sensor_resolution((width, height))
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

    def print_info(self):
        print("==================")
        print(self.cam_config.to_string())
        print("Force USB2: {}".format(self.force_usb2))
        print("==================")

    def get_data(self):
        print("Abstract class Wrapper does not implement get_data()")

    def create_pipeline(self):
        print("Abstract class Wrapper does not implement create_pipeline()")

    def create_queues(self):
        print("Abstract class Wrapper does not implement create_queues()")

    def create_manipRectify(self, pipeline, cam_name, resolution, rectify=True):
        manipRectify = pipeline.createImageManip()
        
        if rectify:
            try:
                mesh, meshWidth, meshHeight = self.get_mesh(cam_name)
            except Exception as e:
                print(e)
                exit()

            manipRectify.setWarpMesh(mesh, meshWidth, meshHeight)

        manipRectify.setMaxOutputFrameSize(resolution[0] * resolution[1] * 3)
        return manipRectify

    def create_manipResize(self, pipeline, resolution):
        manipResize = pipeline.createImageManip()
        manipResize.initialConfig.setResizeThumbnail(resolution[0], resolution[1])
        manipResize.setMaxOutputFrameSize(resolution[0] * resolution[1] * 3)

        # TODO add this in child method for teleop
        # manipResize.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)

        return manipResize

    def compute_undistort_maps(self):
        left_socket = get_socket_from_name("left", self.cam_config.name_to_socket)
        right_socket = get_socket_from_name("right", self.cam_config.name_to_socket)

        resolution = self.cam_config.sensor_resolution

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

    def get_mesh(self, cam_name):
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
    def flash(self, calib_json_file):
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

        print("Opening device ...")
        device = dai.Device()
        print("Flashing ...")
        try:
            device.flashCalibration2(ch)
            print("Calibration flashed successfully")
        except Exception as e:
            print("Flashing failed")
            print(e)
            exit()
