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

    def get_data(
        self,
    ) -> Tuple[Dict[str, npt.NDArray[np.uint8]], Dict[str, float], Dict[str, timedelta]]:
        """Extends the get_data method of the Wrapper class to return the h264 encoded left and right images.

        Returns:
            - Tuple(data, latency, timestamp) : Tuple of dictionaries of h264 encoded left and right images,
                latencies and timestamps for each camera.
        """

        data, latency, ts = super().get_data()

        for name, pkt in data.items():
            data[name] = pkt.getData()

        return data, latency, ts

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

        pipeline = self._create_encoders(pipeline)

        pipeline = self._create_output_streams(pipeline)

        return self._link_pipeline(pipeline)

    def _create_queues(self) -> Dict[str, dai.DataOutputQueue]:
        """Extends the base class method _create_queues() to add the h264 encoded left and right images queues."""

        # config for video: https://docs.luxonis.com/projects/api/en/latest/components/device/#output-queue-maxsize-and-blocking
        queues: Dict[str, dai.DataOutputQueue] = {}
        for name in ["left", "right", "left_mjpeg", "right_mjpeg"]:
            queues[name] = self._device.getOutputQueue(name, maxSize=30, blocking=True)

        return queues
