import json
from typing import Dict, Optional, Tuple

import depthai as dai
import numpy as np
import numpy.typing as npt

from depthai_wrappers.utils import get_socket_from_name


class CamConfig:
    def __init__(
        self,
        cam_config_json: str,
        fps: int,
        resize: Tuple[int, int],
        exposure_params: Tuple[int, int],
        mx_id: str = "",
        isp_scale: Tuple[int, int] = (1, 1),
    ) -> None:
        self.cam_config_json = cam_config_json
        self.fps = fps
        self.exposure_params = exposure_params
        if self.exposure_params is not None:
            assert self.exposure_params[0] is not None and self.exposure_params[1] is not None
            iso = self.exposure_params[1]
            assert 100 <= iso <= 1600

        self.mx_id = mx_id
        self.isp_scale = isp_scale

        config = json.load(open(self.cam_config_json, "rb"))
        self.socket_to_name = config["socket_to_name"]
        self.inverted = config["inverted"]
        self.fisheye = config["fisheye"]
        self.mono = config["mono"]
        self.name_to_socket = {v: k for k, v in self.socket_to_name.items()}
        self.sensor_resolution = (0, 0)
        self.undistort_resolution = (0, 0)
        self.resize_resolution = resize
        self.undstort_maps: Dict[str, Optional[Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]] = {
            "left": None,
            "right": None,
        }
        self.calib: dai.CalibrationHandler = dai.CalibrationHandler()

    def get_device_info(self) -> dai.DeviceInfo:
        return dai.DeviceInfo(self.mx_id)

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
        self.undstort_maps["left"] = (mapXL, mapYL)
        self.undstort_maps["right"] = (mapXR, mapYR)

    def set_calib(self, calib: dai.CalibrationHandler) -> None:
        self.calib = calib

    def get_calib(self) -> dai.CalibrationHandler:
        return self.calib

    def get_K_left(self) -> npt.NDArray[np.float32]:
        left_socket = get_socket_from_name("left", self.name_to_socket)
        left_K = np.array(
            self.calib.getCameraIntrinsics(
                left_socket,
                self.undistort_resolution[0],
                self.undistort_resolution[1],
            )
        )

        return left_K

    def to_string(self) -> str:
        ret_string = "Camera Config: \n"
        ret_string += "FPS: {}\n".format(self.fps)
        ret_string += "Sensor resolution: {}\n".format(self.sensor_resolution)
        ret_string += "Resize resolution: {}\n".format(self.resize_resolution)
        ret_string += "Inverted: {}\n".format(self.inverted)
        ret_string += "Fisheye: {}\n".format(self.fisheye)
        ret_string += "Mono: {}\n".format(self.mono)
        exp = "auto" if self.exposure_params is None else str(self.exposure_params)
        ret_string += "Exposure params: {}\n".format(exp)
        ret_string += "Undistort maps are: " + "set" if self.undstort_maps["left"] is not None else "not set"

        return ret_string
