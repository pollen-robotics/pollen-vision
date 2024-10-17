from datetime import timedelta
from typing import Dict, Optional, Tuple

import depthai as dai
import numpy as np
import numpy.typing as npt
from pollen_vision.camera_wrappers.depthai.wrapper import DepthaiWrapper


class TeleopWrapper(DepthaiWrapper):  # type: ignore[misc]
    """A wrapper for the depthai library that exposes only the relevant features for Pollen's teleoperation feature.

    Calling get_data() returns h264 encoded left and right images.

    Args:
        - cam_config_json: path to the camera configuration json file
        - fps: frames per second
        - force_usb2: force the use of USB2
        - rectify: rectify the images using the calibration data stored in the eeprom of the camera
        - exposure_params: tuple of two integers (exposure, gain) to set the exposure and gain of the camera
        - mx_id: the id of the camera
    """

    def __init__(
        self,
        cam_config_json: str,
        fps: int,
        force_usb2: bool = False,
        rectify: bool = False,
        exposure_params: Optional[Tuple[int, int]] = None,
        mx_id: str = "",
    ) -> None:
        self._data_h264: Dict[str, npt.NDArray[np.uint8]] = {}
        self._latency_h264: Dict[str, float] = {}
        self._ts_h264: Dict[str, timedelta] = {}

        self._data_mjpeg: Dict[str, npt.NDArray[np.uint8]] = {}
        self._latency_mjpeg: Dict[str, float] = {}
        self._ts_mjpeg: Dict[str, timedelta] = {}

        self._queues_mjpeg: Dict[str, dai.DataOutputQueue] = {}

        super().__init__(
            cam_config_json,
            fps,
            force_usb2=force_usb2,
            resize=(960, 720),
            rectify=rectify,
            exposure_params=exposure_params,
            mx_id=mx_id,
            isp_scale=(2, 3),
        )

    def get_data_h264(
        self,
    ) -> Tuple[Dict[str, npt.NDArray[np.uint8]], Dict[str, float], Dict[str, timedelta]]:
        """Extends the get_data method of the Wrapper class to return the h264 encoded left and right images.

        Returns:
            - Tuple(data, latency, timestamp) : Tuple of dictionaries of h264 encoded left and right images,
                latencies and timestamps for each camera.
        """

        for name, queue in self.queues.items():
            pkt = queue.get()
            self._data_h264[name] = pkt.getData()
            self._latency_h264[name] = dai.Clock.now() - pkt.getTimestamp()  # type: ignore[call-arg]
            self._ts_h264[name] = pkt.getTimestamp()

        return self._data_h264, self._latency_h264, self._ts_h264

    def get_data_mjpeg(self) -> Tuple[Dict[str, npt.NDArray[np.uint8]], Dict[str, float], Dict[str, timedelta]]:
        for name, queue in self._queues_mjpeg.items():
            pkt = queue.get()
            self._data_mjpeg[name] = pkt.getData()  # type: ignore[attr-defined]
            self._latency_mjpeg[name] = dai.Clock.now() - pkt.getTimestamp()  # type: ignore[attr-defined, call-arg]
            self._ts_mjpeg[name] = pkt.getTimestamp()  # type: ignore[attr-defined]

        return self._data_mjpeg, self._latency_mjpeg, self._ts_mjpeg

    def _create_output_streams(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        super()._create_output_streams(pipeline)

        self.xout_left_mjpeg = pipeline.createXLinkOut()
        self.xout_left_mjpeg.setStreamName("left_mjpeg")

        self.xout_right_mjpeg = pipeline.createXLinkOut()
        self.xout_right_mjpeg.setStreamName("right_mjpeg")

        return pipeline

    def _link_pipeline(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        """Overloads the base class abstract method to link the pipeline with the nodes together."""

        self.left.isp.link(self.left_manip.inputImage)
        self.left_manip.out.link(self.left_encoder.input)
        self.left_manip.out.link(self.left_encoder_mjpeg.input)
        self.left_encoder.bitstream.link(self.xout_left.input)
        self.right_encoder.bitstream.link(self.xout_right.input)

        self.right.isp.link(self.right_manip.inputImage)
        self.right_manip.out.link(self.right_encoder.input)
        self.right_manip.out.link(self.right_encoder_mjpeg.input)
        self.left_encoder_mjpeg.bitstream.link(self.xout_left_mjpeg.input)
        self.right_encoder_mjpeg.bitstream.link(self.xout_right_mjpeg.input)

        return pipeline

    def _create_encoders(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        """Creates the h264 encoders for the left and right images."""

        profile = dai.VideoEncoderProperties.Profile.H264_BASELINE
        bitrate = 4000
        numBFrames = 0  # no B frames for streaming
        self.left_encoder = pipeline.create(dai.node.VideoEncoder)
        self.left_encoder.setDefaultProfilePreset(self.cam_config.fps, profile)
        self.left_encoder.setKeyframeFrequency(self.cam_config.fps)  # every 1s
        self.left_encoder.setNumBFrames(numBFrames)
        self.left_encoder.setBitrateKbps(bitrate)
        # self.left_encoder.setQuality(self.cam_config.encoder_quality)

        self.right_encoder = pipeline.create(dai.node.VideoEncoder)
        self.right_encoder.setDefaultProfilePreset(self.cam_config.fps, profile)
        self.right_encoder.setKeyframeFrequency(self.cam_config.fps)  # every 1s
        self.right_encoder.setNumBFrames(numBFrames)
        self.right_encoder.setBitrateKbps(bitrate)
        # self.right_encoder.setQuality(self.cam_config.encoder_quality)

        profile = dai.VideoEncoderProperties.Profile.MJPEG

        self.left_encoder_mjpeg = pipeline.create(dai.node.VideoEncoder)
        self.left_encoder_mjpeg.setDefaultProfilePreset(self.cam_config.fps, profile)
        # self.left_encoder_mjpeg.setLossless(True)

        self.right_encoder_mjpeg = pipeline.create(dai.node.VideoEncoder)
        self.right_encoder_mjpeg.setDefaultProfilePreset(self.cam_config.fps, profile)
        # self.right_encoder_mjpeg.setLossless(True)

        return pipeline

    def _create_pipeline(self) -> dai.Pipeline:
        """Creates the pipeline for the depthai device.

        Returns the linked pipeline.
        """

        pipeline = self._pipeline_basis()

        # CAM_B (right) drives by default. Change it to CAM_C (left)
        self.right.initialControl.setMisc("3a-follow", dai.CameraBoardSocket.CAM_C)
        self.left.initialControl.setAutoExposureLimit(10000) # more robust to motion blur
        self.left.initialControl.setMisc("3a-follow", dai.CameraBoardSocket.CAM_C)

        pipeline = self._create_encoders(pipeline)

        pipeline = self._create_output_streams(pipeline)

        return self._link_pipeline(pipeline)

    def _create_queues(self) -> Dict[str, dai.DataOutputQueue]:
        """Extends the base class method _create_queues() to add the h264 encoded left and right images queues."""

        # config for video: https://docs.luxonis.com/projects/api/en/latest/components/device/#output-queue-maxsize-and-blocking
        queues_h264: Dict[str, dai.DataOutputQueue] = {}
        for name in ["left", "right"]:
            queues_h264[name] = self._device.getOutputQueue(name, maxSize=30, blocking=True)

        for name in ["left_mjpeg", "right_mjpeg"]:
            self._queues_mjpeg[name] = self._device.getOutputQueue(name, maxSize=1, blocking=False)

        return queues_h264
