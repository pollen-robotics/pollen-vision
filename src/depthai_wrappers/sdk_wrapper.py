from datetime import timedelta
from typing import Dict, Optional, Tuple

import depthai as dai
import numpy as np
import numpy.typing as npt

from depthai_wrappers.utils import get_socket_from_name
from depthai_wrappers.wrapper import Wrapper


# Depth is left aligned by convention
# TODO do we need to give the option to change this?
class SDKWrapper(Wrapper):  # type: ignore[misc]
    def __init__(
        self,
        cam_config_json: str,
        fps: int = 30,
        force_usb2: bool = False,
        resize: Optional[Tuple[int, int]] = None,
        rectify: bool = False,  # TODO Not working when compute_depth is True for now
        compute_depth: bool = False,
        exposure_params: Optional[Tuple[int, int]] = None,
        mx_id: str = "",
        jpeg_output: bool = False,
    ) -> None:
        self.compute_depth = compute_depth
        self._mjpeg = jpeg_output
        assert not (self.compute_depth and rectify), "Rectify is not working when compute_depth is True for now"

        super().__init__(
            cam_config_json,
            fps,
            force_usb2=force_usb2,
            resize=resize if not compute_depth else (1280, 800),
            rectify=rectify if not compute_depth else False,
            exposure_params=exposure_params,
            mx_id=mx_id,
        )

    def get_data(
        self,
    ) -> Tuple[Dict[str, npt.NDArray[np.uint8]], Dict[str, float], Dict[str, timedelta]]:
        data, latency, ts = super().get_data()
        for name, pkt in data.items():
            data[name] = pkt.getCvFrame()

        return data, latency, ts

    def create_queues(self) -> Dict[str, dai.DataOutputQueue]:
        queues: Dict[str, dai.DataOutputQueue] = super().create_queues()
        if self.compute_depth:
            queues["depth"] = self.device.getOutputQueue("depth", maxSize=1, blocking=False)
            queues["disparity"] = self.device.getOutputQueue("disparity", maxSize=1, blocking=False)

            queues["depthNode_left"] = self.device.getOutputQueue("depthNode_left", maxSize=1, blocking=False)
            queues["depthNode_right"] = self.device.getOutputQueue("depthNode_right", maxSize=1, blocking=False)

        return queues

    def create_output_streams(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        pipeline = super().create_output_streams(pipeline)

        if self.compute_depth:
            self.xout_depth = pipeline.createXLinkOut()
            self.xout_depth.setStreamName("depth")

            self.xout_disparity = pipeline.createXLinkOut()
            self.xout_disparity.setStreamName("disparity")

            self.xout_depthNode_left = pipeline.createXLinkOut()
            self.xout_depthNode_left.setStreamName("depthNode_left")

            self.xout_depthNode_right = pipeline.createXLinkOut()
            self.xout_depthNode_right.setStreamName("depthNode_right")

        return pipeline

    def link_pipeline(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        # Resize, optionally rectify
        self.left.isp.link(self.left_manip.inputImage)
        self.right.isp.link(self.right_manip.inputImage)

        if self._mjpeg:
            self.left_manip.out.link(self.left_encoder.input)
            self.right_manip.out.link(self.right_encoder.input)

            self.left_encoder.bitstream.link(self.xout_left.input)
            self.right_encoder.bitstream.link(self.xout_right.input)
        else:
            self.left_manip.out.link(self.xout_left.input)
            self.right_manip.out.link(self.xout_right.input)

        if self.compute_depth:
            self.left_manip.out.link(self.depth.left)
            self.right_manip.out.link(self.depth.right)

            self.depth.depth.link(self.xout_depth.input)
            self.depth.disparity.link(self.xout_disparity.input)

            self.depth.rectifiedLeft.link(self.xout_depthNode_left.input)
            self.depth.rectifiedRight.link(self.xout_depthNode_right.input)

        return pipeline

    def create_encoders(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        profile = dai.VideoEncoderProperties.Profile.MJPEG
        self.left_encoder = pipeline.create(dai.node.VideoEncoder)
        self.left_encoder.setDefaultProfilePreset(self.cam_config.fps, profile)

        self.right_encoder = pipeline.create(dai.node.VideoEncoder)
        self.right_encoder.setDefaultProfilePreset(self.cam_config.fps, profile)

        return pipeline

    def create_pipeline(self) -> dai.Pipeline:
        pipeline = self.pipeline_basis()

        if self.compute_depth:
            # Configuring depth node
            left_socket = get_socket_from_name("left", self.cam_config.name_to_socket)
            self.depth = pipeline.createStereoDepth()
            self.depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            self.depth.setLeftRightCheck(True)
            self.depth.setExtendedDisparity(False)
            self.depth.setSubpixel(True)
            self.depth.setDepthAlign(left_socket)
            self.depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
            self.depth_max_disparity = self.depth.getMaxDisparity()

            config = self.depth.initialConfig.get()
            config.postProcessing.speckleFilter.enable = False
            config.postProcessing.speckleFilter.speckleRange = 50
            config.postProcessing.temporalFilter.enable = False
            config.postProcessing.spatialFilter.enable = False
            config.postProcessing.spatialFilter.holeFillingRadius = 2
            config.postProcessing.spatialFilter.numIterations = 1
            # config.postProcessing.thresholdFilter.minRange = 400
            # config.postProcessing.thresholdFilter.maxRange = 15000
            # config.postProcessing.decimationFilter.decimationFactor = 1
            self.depth.initialConfig.set(config)

        if self._mjpeg:
            pipeline = self.create_encoders(pipeline)

        pipeline = self.create_output_streams(pipeline)

        return self.link_pipeline(pipeline)
