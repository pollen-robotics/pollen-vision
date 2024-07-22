"""Camera configuration class for depthai cameras."""

import json
from typing import Dict, List, Optional, Tuple

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

    def get_K_left(self) -> npt.NDArray[np.float32]:
        """Returns the intrinsic matrix of the left camera."""
        left_socket = get_socket_from_name("left", self.name_to_socket)
        left_K = np.array(
            self.calib.getCameraIntrinsics(
                left_socket,
                self.undistort_resolution[0],
                self.undistort_resolution[1],
            )
        )

        return left_K

    def get_K_right(self) -> npt.NDArray[np.float32]:
        """Returns the intrinsic matrix of the right camera."""
        right_socket = get_socket_from_name("right", self.name_to_socket)
        right_K = np.array(
            self.calib.getCameraIntrinsics(
                right_socket,
                self.undistort_resolution[0],
                self.undistort_resolution[1],
            )
        )

        return right_K

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

        if side == "left":
            K = self.get_K_left().flatten()
            R = np.array(self.calib.getStereoLeftRectificationRotation()).flatten()
            P_t = np.zeros(3).reshape((3, 1))  # Tx, Ty, 0
            P = np.hstack((self.get_K_left(), P_t)).flatten()

        else:
            K = self.get_K_right().flatten()
            R = np.array(self.calib.getStereoRightRectificationRotation()).flatten()
            Extrinsics = np.array(
                self.calib.getCameraExtrinsics(
                    srcCamera=get_socket_from_name("left", self.name_to_socket),
                    dstCamera=get_socket_from_name("right", self.name_to_socket),
                )
            ).reshape((4, 4))
            P_t = np.zeros(3).reshape((3, 1))  # Tx, Ty, 0
            P_t[0] = Extrinsics[0, 3]  # Tx
            P_t[1] = Extrinsics[1, 3]  # Ty
            P = np.hstack((self.get_K_right(), P_t)).flatten()

        return height, width, distortion_model, D, K, R, P
