from datetime import timedelta
from typing import Dict, Optional, Tuple
from pollen_vision.camera_wrappers import CameraWrapper
import cv2
import numpy as np
import numpy.typing as npt
from pypylon import pylon


class BaslerWrapper(CameraWrapper):
    def __init__(self) -> None:

        self.devices = pylon.TlFactory.GetInstance().EnumerateDevices()
        self.cameras = pylon.InstantCameraArray(min(len(self.devices), 2))
        # Setup converters for each camera
        self.converters = []
        self.cameras_names = ["left", "right"]

        for i, camera in enumerate(self.cameras):
            camera.Attach(pylon.TlFactory.GetInstance().CreateDevice(self.devices[i]))
            camera.Open()

            # Camera Configuration
            camera.PixelFormat.Value = "BayerRG8"
            camera.Width.Value = camera.Width.Max
            camera.Height.Value = camera.Height.Max
            camera.AcquisitionFrameRateEnable.Value = True
            camera.AcquisitionFrameRate.Value = 50.0
            camera.DeviceLinkThroughputLimitMode = "Off"

            # Setup a converter for each camera
            converter = pylon.ImageFormatConverter()
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            self.converters.append(converter)

        self.cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        self.K = np.eye(3).astype(np.float32)
        self.D = np.zeros(5).astype(np.float32)

    def get_data(self) -> Tuple[Dict[str, npt.NDArray[np.uint8]], Optional[Dict[str, float]], Optional[Dict[str, timedelta]]]:

        data: Dict[str, npt.NDArray[np.uint8]] = {}
        latency: Dict[str, float] = {}
        ts: Dict[str, timedelta] = {}

        for i, camera in enumerate(self.cameras):
            camera_name = self.cameras_names[i]
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grabResult.GrabSucceeded():
                img_rgb = self.converters[i].Convert(grabResult).GetArray()

                data[camera_name] = img_rgb
                latency[camera_name] = 0  # TODO
                ts[camera_name] = timedelta(0)  # TODO

            grabResult.Release()

        return data, latency, ts

    def get_K(self) -> npt.NDArray[np.float32]:
        return self.K

    def get_D(self) -> npt.NDArray[np.float32]:
        return self.D


if __name__ == "__main__":
    camera = BaslerWrapper()
    while True:
        data, latency, ts = camera.get_data()
        for camera_name, img in data.items():
            cv2.imshow(camera_name, img)
        cv2.waitKey(1)
