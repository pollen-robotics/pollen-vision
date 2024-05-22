from pollen_vision.camera_wrappers import CameraWrapper
from datetime import timedelta
from typing import Dict, Tuple
import time
import cv2
import numpy as np
import numpy.typing as npt

from reachy2_sdk import ReachySDK

class PollenSDKCameraWrapper(CameraWrapper):
    def __init__(self, robot: ReachySDK, cam: str="SR") -> None:
        super().__init__()
        self._reachy=robot
        self._cam_name=cam
        try:
            self._reachy.connect()
            time.sleep(1)
            self._logger.info("Connected to Reachy")
        except Exception as err:
            self._logger.error(f"Cannot connect to Reachy: {err}")
            raise err

        self.depth=None
        self.left=None
        self.right=None

    def get_data(self) -> Tuple[Dict[str, npt.NDArray[np.uint8]], Dict[str, float], Dict[str, timedelta]]:

        data={}
        latency={}
        ts={}
        try:
            if not self._reachy.is_connected():
                self._reachy.connect()

            cam=getattr(self._reachy.cameras, self._cam_name)
            if cam.capture():
                self.depth=cam.get_depthmap()
                data["depth"]=self.depth
                self.left=cam.get_frame()
                data["left"]=self.left
                # self.right=cam.get_frame() #FIXME, for now, can't get the RIGHT image from the sdk...
                # data["right"]=self.left

                return data,latency,ts
            else:
                self._logger.error(f"capture failed")
                return data,latency,ts
        except Exception as err:
            self._logger.error(f"Cannot capture frame: {err}")
            raise err

    def get_K(self, left: bool = True) -> npt.NDArray[np.float32]:
        intrinsics={}
        try:
            if not self._reachy.is_connected():
                self._reachy.connect()

            cam=getattr(self._reachy.cameras, self._cam_name)

            if cam.capture():

                intrinsics["left"]=cam.get_intrinsic_matrix()
                intrinsics["depth"]=cam.get_depth_intrinsic_matrix()
            else:
                self._logger.error(f"capture failed")

            return intrinsics

        except Exception as err:
            self._logger.error(f"Cannot get instrinsic: {err}")
            raise err


    def get_depth_K(self) -> npt.NDArray[np.float32]:
        intrinsics={}
        try:
            if not self._reachy.is_connected():
                self._reachy.connect()

            cam=getattr(self._reachy.cameras, self._cam_name)
            if cam.capture():
                intrinsics["depth"]=cam.get_depth_intrinsic_matrix()
            else:
                self._logger.error(f"capture failed")

            return intrinsics


        except Exception as err:
            self._logger.error(f"Cannot get instrinsic: {err}")
            raise err


if __name__ == '__main__':
    reachy=ReachySDK('localhost')
    sdkcam=PollenSDKCameraWrapper(reachy)
    data,_,_=sdkcam.get_data()
    print(data)
    K=sdkcam.get_K()
    print(K)

    dK=sdkcam.get_depth_K()
    print(dK)
