import time
from datetime import timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
from pollen_vision.camera_wrappers import CameraWrapper
from reachy2_sdk import ReachySDK  # noqa: F401
from reachy2_sdk.media.camera import CameraView  # noqa: F401


class PollenSDKCameraWrapper(CameraWrapper):  # type: ignore[misc]
    def __init__(self, robot: ReachySDK, cam: str = "SR") -> None:
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
        # self.depthleft=None
        # self.depthright=None

    def get_data(self) -> Tuple[Dict[str, npt.NDArray[np.uint8]], Dict[str, float], Dict[str, timedelta]]:
        data: Dict[str, npt.NDArray[np.uint8]] = {}
        latency: Dict[str, float] = {}
        ts: Dict[str, timedelta] = {}

        try:
            if not self._reachy.is_connected():
                self._reachy.connect()

            cam = getattr(self._reachy.cameras, self._cam_name)
            if cam.capture():
                self.depth = cam.get_depthmap()
                data["depth"] = self.depth  # type: ignore
                self.left = cam.get_frame()
                data["left"] = self.left  # type: ignore
                # self.right=cam.get_frame() #FIXME, for now, can't get the RIGHT image from the sdk...
                # data["right"]=self.left
                self.right = cam.get_frame(CameraView.RIGHT)
                data["right"] = self.right  # type: ignore
                # fixme
                data["depthNode_left"] = self.left  # type: ignore
                data["depthNode_right"] = self.right  # type: ignore

                return data, latency, ts
            else:
                self._logger.error("capture failed")
                return data, latency, ts
        except Exception as err:
            self._logger.error(f"Cannot capture frame: {err}")
            raise err

    def get_K(self, left: bool = True) -> Optional[npt.NDArray[np.float32]]:
        try:
            if not self._reachy.is_connected():
                self._reachy.connect()

            cam = getattr(self._reachy.cameras, self._cam_name)

            if not cam.capture():
                self._logger.error("capture failed")
                return None
            # intrinsics["left"]=cam.get_intrinsic_matrix()
            # intrinsics["depth"]=cam.get_depth_intrinsic_matrix()

            # always left... FIXME
            return cam.get_intrinsic_matrix()  # type: ignore

        except Exception as err:
            self._logger.error(f"Cannot get instrinsic: {err}")
            raise err

    def get_depth_K(self) -> Optional[npt.NDArray[np.float32]]:
        try:
            if not self._reachy.is_connected():
                self._reachy.connect()

            cam = getattr(self._reachy.cameras, self._cam_name)
            if not cam.capture():
                self._logger.error("capture failed")
                return None

            # intrinsics["depth"]=cam.get_depth_intrinsic_matrix()
            return cam.get_depth_intrinsic_matrix()  # type: ignore

        except Exception as err:
            self._logger.error(f"Cannot get instrinsic: {err}")
            raise err


# if __name__ == "__main__":
#     reachy = ReachySDK("localhost")
#     sdkcam = PollenSDKCameraWrapper(reachy)
#     data, _, _ = sdkcam.get_data()  # type: ignore
#     print(data)
#     K = sdkcam.get_K()
#     print(K)

#     dK = sdkcam.get_depth_K()
#     print(dK)
