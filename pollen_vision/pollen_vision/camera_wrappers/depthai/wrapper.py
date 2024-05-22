"""Depthai Wrapper module.
"""

import json
import os
from abc import abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

import depthai as dai
import numpy as np
import numpy.typing as npt
from pollen_vision.camera_wrappers import CameraWrapper
from pollen_vision.camera_wrappers.depthai.calibration.undistort import (
    compute_undistort_maps,
    get_mesh,
)
from pollen_vision.camera_wrappers.depthai.cam_config import CamConfig
from pollen_vision.camera_wrappers.depthai.utils import (
    get_inv_R_T,
    get_socket_from_name,
    socket_camToString,
)


class DepthaiWrapper(CameraWrapper):  # type: ignore
    """Wrapper is an abstract class for luxonis cameras using the depthai library.

    It factors out the common code between the different camera wrappers.
    """

    def __init__(
        self,
        cam_config_json: str,
        fps: int,
        force_usb2: bool,
        resize: Tuple[int, int],
        rectify: bool,
        exposure_params: Tuple[int, int],
        mx_id: str,
        isp_scale: Tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        self.cam_config = CamConfig(cam_config_json, fps, resize, exposure_params, mx_id, isp_scale, rectify, force_usb2)
        # self._logger = logging.getLogger(__name__)

        self._prepare()

    def _prepare(self) -> None:
        """Prepares the camera for use.

        Sets up :
        - camera configuration
        - device connection
        - pipeline
        - queues

        If requested, pre-computes the undistort maps for the rectification.

        """
        self._logger.debug("Connecting to camera")

        self._device = dai.Device(
            self.cam_config.get_device_info(),
            maxUsbSpeed=(dai.UsbSpeed.HIGH if self.cam_config.force_usb2 else dai.UsbSpeed.SUPER_PLUS),
        )

        connected_cameras_features = []
        for cam in self._device.getConnectedCameraFeatures():
            if socket_camToString[cam.socket] in self.cam_config.socket_to_name.keys():
                connected_cameras_features.append(cam)

        # Assuming both cameras are the same
        width = connected_cameras_features[0].width
        height = connected_cameras_features[0].height

        self.cam_config.set_sensor_resolution((width, height))

        # Note: doing this makes the teleopWrapper not work with cams other than the teleoperation head.
        # This comes from the (2, 3) ispscale factor that is not appropriate for 1280x800 resolution.
        # Not really a big deal
        width_undistort_resolution = int(width * (self.cam_config.isp_scale[0] / self.cam_config.isp_scale[1]))
        height_unistort_resolution = int(height * (self.cam_config.isp_scale[0] / self.cam_config.isp_scale[1]))
        self.cam_config.set_undistort_resolution((width_undistort_resolution, height_unistort_resolution))
        self.cam_config.set_calib(self._device.readCalibration())

        if self.cam_config.rectify:
            self._set_undistort_maps()

        self.pipeline = self._create_pipeline()

        self._device.startPipeline(self.pipeline)
        self.queues = self._create_queues()

        self.print_info()

    def print_info(self) -> None:
        """Prints the camera configuration."""
        self._logger.info(self.cam_config.to_string())

    def get_data(
        self,
    ) -> Tuple[Dict[str, npt.NDArray[np.uint8]], Dict[str, float], Dict[str, timedelta]]:
        """Gets the data from the camera.

        Returns a tuple containing the data, the latency and the timestamp.
        data is a dict containing the left and right images as well as the depth map if it exists.
        The content of data is defined by the queues created in the _create_queues method.
        """

        data: Dict[str, npt.NDArray[np.uint8]] = {}
        latency: Dict[str, float] = {}
        ts: Dict[str, timedelta] = {}

        for name, queue in self.queues.items():
            pkt = queue.get()
            data[name] = pkt  # type: ignore[assignment]
            latency[name] = dai.Clock.now() - pkt.getTimestamp()  # type: ignore[attr-defined, call-arg]
            ts[name] = pkt.getTimestamp()  # type: ignore[attr-defined]

        return data, latency, ts

    def get_K(self, left: bool = True) -> npt.NDArray[np.float32]:
        return self.cam_config.get_K_left() if left else self.cam_config.get_K_right()  # type: ignore[no-any-return]

    @abstractmethod
    def _create_pipeline(self) -> dai.Pipeline:
        """Abstract method that is implemented by the subclasses."""

        self._logger.error("Abstract class DepthaiWrapper does not implement create_pipeline()")
        exit()

    def _pipeline_basis(self) -> dai.Pipeline:
        """Creates and configures the left and right cameras and the image manip nodes.

        This method is used (and/or extended) by the subclasses to create the basis pipeline.
        """

        self._logger.debug("Configuring depthai pipeline")
        pipeline = dai.Pipeline()

        left_socket = get_socket_from_name("left", self.cam_config.name_to_socket)
        right_socket = get_socket_from_name("right", self.cam_config.name_to_socket)

        self.left = pipeline.createColorCamera()
        self.left.setFps(self.cam_config.fps)
        self.left.setBoardSocket(left_socket)
        self.left.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1440X1080)
        self.left.setIspScale(*self.cam_config.isp_scale)

        self.right = pipeline.createColorCamera()
        self.right.setFps(self.cam_config.fps)
        self.right.setBoardSocket(right_socket)
        self.right.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1440X1080)
        self.right.setIspScale(*self.cam_config.isp_scale)

        # self.cam_config.set_undistort_resolution(self.left.getIspSize())

        if self.cam_config.exposure_params is not None:
            self.left.initialControl.setManualExposure(*self.cam_config.exposure_params)
            self.right.initialControl.setManualExposure(*self.cam_config.exposure_params)
        if self.cam_config.inverted:
            self.left.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
            self.right.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)

        self.left_manip = self._create_imageManip(
            pipeline, "left", self.cam_config.undistort_resolution, self.cam_config.rectify
        )
        self.right_manip = self._create_imageManip(
            pipeline, "right", self.cam_config.undistort_resolution, self.cam_config.rectify
        )

        return pipeline

    @abstractmethod
    def _link_pipeline(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        """Abstract method that is implemented by the subclasses.
        Links the nodes together.
        """

        self._logger.error("Abstract class DepthaiWrapper does not implement link_pipeline()")
        exit()

    def _create_output_streams(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        """Creates and names the output streams.
        This method is used (and/or extended) by the subclasses.
        """

        self.xout_left = pipeline.createXLinkOut()
        self.xout_left.setStreamName("left")

        self.xout_right = pipeline.createXLinkOut()
        self.xout_right.setStreamName("right")

        return pipeline

    def _create_queues(self) -> Dict[str, dai.DataOutputQueue]:
        """Creates the output queues.
        This method is used (and/or extended) by the subclasses.
        """
        queues: Dict[str, dai.DataOutputQueue] = {}
        for name in ["left", "right"]:
            queues[name] = self._device.getOutputQueue(name, maxSize=1, blocking=False)
        return queues

    def _create_imageManip(
        self,
        pipeline: dai.Pipeline,
        cam_name: str,
        resolution: Tuple[int, int],
        rectify: bool = True,
    ) -> dai.node.ImageManip:
        """Resize and optionally rectify an image"""

        manip = pipeline.createImageManip()

        if rectify:
            try:
                mesh, meshWidth, meshHeight = get_mesh(self.cam_config, cam_name)
                manip.setWarpMesh(mesh, meshWidth, meshHeight)
            except Exception as e:
                self._logger.error(e)
                exit()
        manip.setMaxOutputFrameSize(resolution[0] * resolution[1] * 3)

        manip.initialConfig.setKeepAspectRatio(True)
        manip.initialConfig.setResize(resolution[0], resolution[1])
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)

        return manip

    def _set_undistort_maps(self) -> None:
        """Computes and assign the undistort maps for the rectification."""
        mapXL, mapYL, mapXR, mapYR = compute_undistort_maps(self.cam_config)
        self.cam_config.set_undistort_maps(mapXL, mapYL, mapXR, mapYR)

    # Takes in the output of multical calibration
    def flash(self, calib_json_file: str) -> None:
        """Flashes the calibration to the camera.

        The calibration is read from the calib_json_file and flashed into the camera's eeprom.
        """
        now = str(datetime.now()).replace(" ", "_").split(".")[0]

        device_calibration_backup_file = Path("./CALIBRATION_BACKUP_" + now + ".json")
        deviceCalib = self._device.readCalibration()
        deviceCalib.eepromToJsonFile(device_calibration_backup_file)
        self._logger.info(f"Backup of device calibration saved to {device_calibration_backup_file}")

        os.environ["DEPTHAI_ALLOW_FACTORY_FLASHING"] = "235539980"

        ch = dai.CalibrationHandler()
        calibration_data = json.load(open(calib_json_file, "rb"))

        cameras = calibration_data["cameras"]
        camera_poses = calibration_data["camera_poses"]

        self._logger.info("Setting intrinsics ...")
        for cam_name, params in cameras.items():
            K = np.array(params["K"])
            D = np.array(params["dist"]).reshape((-1))
            im_size = params["image_size"]
            cam_socket = get_socket_from_name(cam_name, self.cam_config.name_to_socket)

            ch.setCameraIntrinsics(cam_socket, K.tolist(), im_size)
            ch.setDistortionCoefficients(cam_socket, D.tolist())
            if self.cam_config.fisheye:
                self._logger.info("Setting camera type to fisheye ...")
                ch.setCameraType(cam_socket, dai.CameraModel.Fisheye)

        self._logger.info("Setting extrinsics ...")
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
            R_right_to_left.tolist(),
            T_right_to_left.tolist(),
            specTranslation=T_right_to_left.tolist(),
        )
        ch.setCameraExtrinsics(
            right_socket,
            left_socket,
            R_left_to_right,
            T_left_to_right,
            specTranslation=T_left_to_right,
        )

        ch.setStereoLeft(left_socket, np.eye(3).tolist())
        ch.setStereoRight(right_socket, R_right_to_left.tolist())

        self._logger.info("Flashing ...")
        try:
            self._device.flashCalibration2(ch)
            self._logger.info("Calibration flashed successfully")
        except Exception as e:
            self._logger.error("Flashing failed")
            self._logger.error(e)
            exit()


if __name__ == '__main__':
    from pollen_vision.camera_wrappers.depthai.utils import get_config_file_path, get_connected_devices
    from pollen_vision.camera_wrappers.depthai import SDKWrapper

    devices = get_connected_devices()
    print(f'Detected cameras: {devices}')
    for mxid, name in devices.items():

        if name == 'other':
            cam = SDKWrapper(
                get_config_file_path("CONFIG_SR"),
                compute_depth=True,
                rectify=False,
                mx_id=mxid,
                jpeg_output=True,
            )
        else:
            cam = SDKWrapper(
                get_config_file_path("CONFIG_IMX296"),
                compute_depth=False,
                rectify=True,
                mx_id=mxid,
                jpeg_output=True,
            )
