"""Camera configuration class for depthai cameras."""

import json
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import depthai as dai
import numpy as np
import numpy.typing as npt
from pollen_vision.camera_wrappers.depthai.utils import get_socket_from_name


class CamConfig:
    """CamConfig handles some of the camera configuration for depthai cameras.

    Non self explanatory parameters:
    - exposure_params: a tuple of two integers, the first one is the exposure time in microseconds,
                       the second one is the ISO value. Default means auto exposure.
    - isp_scale: a tuple of two integers. The first one is the numerator
                 and the second one is the denominator of the scale factor.
                 This is used to scale the image signal processor (ISP) output.
    - mx_id: the mx_id of the device.
             This allows connecting to multiple devices plugged in the host machine at the same time,
             differentiating them by their mx_id.
    - force_usb2: if True, forces the camera to use USB2 instead of USB3.

    """

    def __init__(
        self,
        cam_config_json: str,
        fps: int,
        resize: Tuple[int, int],
        exposure_params: Tuple[int, int],
        mx_id: str = "",
        isp_scale: Tuple[int, int] = (1, 1),
        rectify: bool = False,
        force_usb2: bool = False,
        encoder_quality: int = 80,
    ) -> None:
        self._cam_config_json = cam_config_json
        self.fps = fps
        self.exposure_params = exposure_params
        if self.exposure_params is not None:
            assert self.exposure_params[0] is not None and self.exposure_params[1] is not None
            iso = self.exposure_params[1]
            assert 100 <= iso <= 1600

        self._mx_id = mx_id
        self.isp_scale = isp_scale
        self.rectify = rectify
        self.force_usb2 = force_usb2
        self.encoder_quality = encoder_quality

        config = json.load(open(self._cam_config_json, "rb"))
        self.socket_to_name = config["socket_to_name"]
        self.inverted = config["inverted"]
        self.fisheye = config["fisheye"]
        self.mono = config["mono"]
        self.name_to_socket = {v: k for k, v in self.socket_to_name.items()}
        self.sensor_resolution = (0, 0)
        self.undistort_resolution = (0, 0)
        self.resize_resolution = resize
        self.undistort_maps: Dict[str, Optional[Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]] = {
            "left": None,
            "right": None,
        }
        self.calib: dai.CalibrationHandler = dai.CalibrationHandler()

        # lazy init, camera needs to be connected to
        self.P_left: Optional[npt.NDArray[np.float32]] = None
        self.P_right: Optional[npt.NDArray[np.float32]] = None

        self._logger = logging.getLogger(__name__)

    def get_device_info(self) -> dai.DeviceInfo:
        """Returns a dai.DeviceInfo object with the mx_id.
        This allows connecting to multiple devices plugged in the host machine at the same time,
        differentiating them by their mx_id.
        """

        return dai.DeviceInfo(self._mx_id)

    def set_sensor_resolution(self, resolution: Tuple[int, int]) -> None:
        self.sensor_resolution = resolution

        # Assuming that the resize resolution is the same as the sensor resolution until set otherwise
        if self.resize_resolution is None:
            self.resize_resolution = resolution

    def set_undistort_resolution(self, resolution: Tuple[int, int]) -> None:
        self.undistort_resolution = resolution

    def set_resize_resolution(self, resolution: Tuple[int, int]) -> None:
        self.resize_resolution = resolution

    def set_undistort_maps(
        self,
        mapXL: npt.NDArray[np.float32],
        mapYL: npt.NDArray[np.float32],
        mapXR: npt.NDArray[np.float32],
        mapYR: npt.NDArray[np.float32],
    ) -> None:
        self.undistort_maps["left"] = (mapXL, mapYL)
        self.undistort_maps["right"] = (mapXR, mapYR)

    def set_calib(self, calib: dai.CalibrationHandler) -> None:
        self.calib = calib

    def get_calib(self) -> dai.CalibrationHandler:
        """Returns a dai.CalibrationHandler object with all the camera's calibration data."""
        return self.calib

    def check_cam_name_available(self, cam_name: str) -> None:
        if cam_name not in list(self.name_to_socket.keys()):
            raise ValueError(
                f"Camera {cam_name} not found in the config. Available cameras: {list(self.name_to_socket.keys())}"
            )

    def get_K(self, cam_name: str = "left") -> npt.NDArray[np.float32]:
        """Returns the intrinsic matrix of the requested camera."""
        self.check_cam_name_available(cam_name)

        socket = get_socket_from_name(cam_name, self.name_to_socket)
        K = np.array(
            self.calib.getCameraIntrinsics(
                socket,
                self.undistort_resolution[0],
                self.undistort_resolution[1],
            )
        )

        return K

    def get_D(self, cam_name: str = "left") -> npt.NDArray[np.float32]:
        self.check_cam_name_available(cam_name)

        socket = get_socket_from_name(cam_name, self.name_to_socket)

        D = np.array(self.calib.getDistortionCoefficients(socket))
        return D

    def get_K_left(self) -> npt.NDArray[np.float32]:
        """Returns the intrinsic matrix of the left camera."""
        self._logger.warning('This function is deprecated. Use get_K("left")')

        return self.get_K("left")

    def get_K_right(self) -> npt.NDArray[np.float32]:
        """Returns the intrinsic matrix of the right camera."""
        self._logger.warning('This function is deprecated. Use get_K("right")')

        return self.get_K("right")

    def to_string(self) -> str:
        ret_string = "Camera Config: \n"
        ret_string += "FPS: {}\n".format(self.fps)
        ret_string += "Sensor resolution: {}\n".format(self.sensor_resolution)
        ret_string += "Resize resolution: {}\n".format(self.resize_resolution)
        ret_string += "Inverted: {}\n".format(self.inverted)
        ret_string += "Fisheye: {}\n".format(self.fisheye)
        ret_string += "Mono: {}\n".format(self.mono)
        ret_string += "MX ID: {}\n".format(self._mx_id)
        ret_string += "rectify: {}\n".format(self.rectify)
        ret_string += "force_usb2: {}\n".format(self.force_usb2)
        exp = "auto" if self.exposure_params is None else str(self.exposure_params)
        ret_string += "Exposure params: {}\n".format(exp)
        ret_string += "Undistort maps are: " + "set" if self.undistort_maps["left"] is not None else "not set"

        return ret_string

    def compute_projection_matrices(self) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        left_socket = get_socket_from_name("left", self.name_to_socket)
        right_socket = get_socket_from_name("right", self.name_to_socket)

        left_D = np.array(self.calib.getDistortionCoefficients(left_socket))
        right_D = np.array(self.calib.getDistortionCoefficients(right_socket))

        R = np.array(self.calib.getStereoRightRectificationRotation())

        T = np.array(self.calib.getCameraTranslationVector(left_socket, right_socket))
        T *= 0.01  # to meter for ROS

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.get_K_left(),
            left_D,
            self.get_K_right(),
            right_D,
            self.undistort_resolution,
            R,
            T,
            flags=0,
        )
        return P1.astype(np.float32), P2.astype(np.float32)

    def to_ROS_msg(
        self, side: str = "left"
    ) -> Tuple[int, int, str, List[float], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        # as defined in https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html

        height = self.resize_resolution[1]
        width = self.resize_resolution[0]
        distortion_model = "plumb_bob"
        if self.calib.getDistortionModel(get_socket_from_name(side, self.name_to_socket)) == dai.CameraModel.Fisheye:
            distortion_model = "equidistant"
        D = self.calib.getDistortionCoefficients(get_socket_from_name(side, self.name_to_socket))

        if self.P_left is None or self.P_right is None:
            self.P_left, self.P_right = self.compute_projection_matrices()

        if side == "left":
            K = self.get_K_left().flatten()
            R = np.array(self.calib.getStereoLeftRectificationRotation()).flatten()
            P = np.array(self.P_left).flatten()

        else:
            K = self.get_K_right().flatten()
            R = np.array(self.calib.getStereoRightRectificationRotation()).flatten()
            P = np.array(self.P_right).flatten()

        return height, width, distortion_model, D, K, R, P
