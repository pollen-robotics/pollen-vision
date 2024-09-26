import time
from datetime import timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
from pollen_vision.camera_wrappers import CameraWrapper
from pyquaternion import Quaternion as pyQuat
from reachy2_sdk import ReachySDK  # noqa: F401
from reachy2_sdk.media.camera import CameraView  # noqa: F401


class PollenSDKCameraWrapper(CameraWrapper):  # type: ignore[misc]
    def __init__(self, robot: ReachySDK, cam: str = "depth") -> None:
        super().__init__()

        self._reachy = robot
        self._cam_name = cam
        try:
            self._reachy.connect()
            time.sleep(1)
            self._logger.info("Connected to Reachy")
        except Exception as err:
            self._logger.error(f"Cannot connect to Reachy: {err}")
            raise err

        self.depth = None
        self.left = None
        self.right = None

    def get_data(self) -> Tuple[Dict[str, npt.NDArray[np.uint8]], Dict[str, float], Dict[str, timedelta]]:
        data: Dict[str, npt.NDArray[np.uint8]] = {}
        latency: Dict[str, float] = {}
        ts: Dict[str, timedelta] = {}

        try:
            if not self._reachy.is_connected():
                self._reachy.connect()

            if self._cam_name == "depth":
                frame, ts = self._reachy.cameras.depth.get_frame()
                depth_frame, ts_depth = self._reachy.cameras.depth.get_depth_frame()
                data["depth"] = depth_frame
            if self._cam_name == "teleop":
                frame, ts = self._reachy.cameras.teleop.get_frame()

            data["left"] = frame[:, :, ::-1]

            return data, latency, ts

        except Exception as err:
            self._logger.error(f"Cannot capture frame: {err}")
            raise err

    def get_K(self, left: bool = True) -> Optional[npt.NDArray[np.float32]]:
        try:
            if not self._reachy.is_connected():
                self._reachy.connect()

            if self._cam_name == "depth":
                return np.array(self._reachy.cameras.depth.get_parameters()[4]).reshape(3, 3)
            elif self._cam_name == "teleop":
                return np.array(self._reachy.cameras.teleop.get_parameters()[4]).reshape(3, 3)
            else:
                self._logger.error("Unknown camera")
                return None

        except Exception as err:
            self._logger.error(f"Cannot get intrinsic: {err}")
            raise err

    def get_depth_K(self) -> Optional[npt.NDArray[np.float32]]:
        try:
            if not self._reachy.is_connected():
                self._reachy.connect()

            cam = getattr(self._reachy.cameras, self._cam_name)
            if not cam.capture():
                self._logger.error("capture failed")
                return None

            return cam.get_depth_intrinsic_matrix()  # type: ignore

        except Exception as err:
            self._logger.error(f"Cannot get instrinsic: {err}")
            raise err

    def get_head_orientation(self) -> pyQuat:
        try:
            if not self._reachy.is_connected():
                self._reachy.connect()

            return self._reachy.head.get_orientation()

        except Exception as err:
            self._logger.error(f"Cannot get head orientation: {err}")
            raise err

    @property
    def cam_name(self) -> str:
        return self._cam_name


# if __name__ == "__main__":
#     reachy = ReachySDK("localhost")
#     sdkcam = PollenSDKCameraWrapper(reachy)
#     data, _, _ = sdkcam.get_data()  # type: ignore
#     print(data)
#     K = sdkcam.get_K()
#     print(K)

#     dK = sdkcam.get_depth_K()
#     print(dK)
