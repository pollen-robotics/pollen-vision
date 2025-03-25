from datetime import timedelta
from typing import Dict, Optional, Tuple
from pollen_vision.camera_wrappers import CameraWrapper
import cv2
import numpy as np
import numpy.typing as npt
from pypylon import pylon
import json


class BaslerWrapper(CameraWrapper):
    def __init__(self, undistort: bool = False, calib_file_path: str = "calibration/calib_images/calibration.json") -> None:
        self.undistort = undistort
        if self.undistort:
            try:
                self.calibration_data = json.load(open(calib_file_path, "r"))
            except Exception as e:
                print(e)
                exit()

        self.devices = pylon.TlFactory.GetInstance().EnumerateDevices()
        self.cameras = pylon.InstantCameraArray(min(len(self.devices), 2))

        self.converters = []
        self.cameras_names = ["right", "left"]  # TODO is this repetable ? probably not

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

        self.K = {"left": np.eye(3), "right": np.eye(3)}
        self.D = {"left": np.zeros(5), "right": np.zeros(5)}
        self.resolution = None
        if self.undistort:
            cameras = self.calibration_data["cameras"]
            camera_poses = self.calibration_data["camera_poses"]

            for cam_name, params in cameras.items():
                if self.resolution is None:
                    self.resolution = (params["image_size"][0], params["image_size"][1])
                self.K[cam_name] = np.array(params["K"])
                self.D[cam_name] = np.array(params["dist"])

            self.right_to_left = camera_poses["right_to_left"]
            self.R_right_to_left = np.array(self.right_to_left["R"])
            self.T_right_to_left = np.array(self.right_to_left["T"])

            self.mapXL, self.mapYL, self.mapXR, self.mapYR = self.compute_undistort_maps()

    def compute_undistort_maps(self) -> Tuple[cv2.UMat, cv2.UMat, cv2.UMat, cv2.UMat]:
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.K["left"],
            self.D["left"],
            self.K["right"],
            self.D["right"],
            self.resolution,
            self.R_right_to_left,
            self.T_right_to_left,
            flags=0,
        )

        mapXL, mapYL = cv2.fisheye.initUndistortRectifyMap(self.K["left"], self.D["left"], R1, P1, self.resolution, 5)
        mapXR, mapYR = cv2.fisheye.initUndistortRectifyMap(self.K["right"], self.D["right"], R2, P2, self.resolution, 5)

        return mapXL, mapYL, mapXR, mapYR

    def get_data(self) -> Tuple[Dict[str, npt.NDArray[np.uint8]], Optional[Dict[str, float]], Optional[Dict[str, timedelta]]]:

        data: Dict[str, npt.NDArray[np.uint8]] = {}
        latency: Dict[str, float] = {}
        ts: Dict[str, timedelta] = {}

        for i, camera in enumerate(self.cameras):
            camera_name = self.cameras_names[i]
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grabResult.GrabSucceeded():
                img_rgb = self.converters[i].Convert(grabResult).GetArray()

                if self.undistort:
                    if camera_name == "left":
                        img_rgb = cv2.remap(img_rgb, self.mapXL, self.mapYL, cv2.INTER_LINEAR)
                    elif camera_name == "right":
                        img_rgb = cv2.remap(img_rgb, self.mapXR, self.mapYR, cv2.INTER_LINEAR)

                data[camera_name] = img_rgb
                latency[camera_name] = 0  # TODO
                ts[camera_name] = timedelta(0)  # TODO

            grabResult.Release()

        if len(data.keys()) != 2:
            raise Exception("Could not retrieve both images")

        return data, latency, ts

    def get_K(self, cam_name: str = "left") -> npt.NDArray[np.float32]:
        return self.K[cam_name]

    def get_D(self, cam_name: str = "left") -> npt.NDArray[np.float32]:
        return self.D[cam_name]


if __name__ == "__main__":
    camera = BaslerWrapper(undistort=True)
    while True:
        try:
            data, latency, ts = camera.get_data()
        except Exception as e:
            print(e)
            continue

        concat = np.hstack((data["left"], data["right"]))
        cv2.imshow("concat", cv2.resize(concat, (0, 0), fx=0.5, fy=0.5))
        cv2.waitKey(1)
