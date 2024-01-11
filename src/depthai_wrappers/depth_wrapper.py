from datetime import timedelta
from typing import Dict, Optional, Tuple

import depthai as dai
import numpy as np
import numpy.typing as npt

from depthai_wrappers.utils import get_socket_from_name
from depthai_wrappers.wrapper import Wrapper


class DepthWrapper(Wrapper):  # type: ignore[misc]
    def __init__(
        self,
        cam_config_json: str,
        fps: int,
        force_usb2: bool = False,
        exposure_params: Optional[Tuple[int, int]] = None,
        mx_id: str = "",
    ) -> None:
        super().__init__(
            cam_config_json,
            fps,
            force_usb2=force_usb2,
            resize=(1280, 800),
            rectify=False,
            exposure_params=exposure_params,
            mx_id=mx_id,
        )

    def get_data(
        self,
    ) -> Tuple[Dict[str, npt.NDArray[np.uint8]], Dict[str, float], Dict[str, timedelta],]:
        data, latency, ts = super().get_data()
        for name, pkt in data.items():
            data[name] = pkt.getCvFrame()

        return data, latency, ts

    def create_queues(self) -> Dict[str, dai.DataOutputQueue]:
        queues: Dict[str, dai.DataOutputQueue] = super().create_queues()
        queues["depth"] = self.device.getOutputQueue("depth", maxSize=1, blocking=False)
        queues["disparity"] = self.device.getOutputQueue("disparity", maxSize=1, blocking=False)
        return queues

    def create_output_streams(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        pipeline = super().create_output_streams(pipeline)

        self.xout_depth = pipeline.createXLinkOut()
        self.xout_depth.setStreamName("depth")

        self.xout_disparity = pipeline.createXLinkOut()
        self.xout_disparity.setStreamName("disparity")

        return pipeline

    def link_pipeline(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        self.left.isp.link(self.left_manipRescale.inputImage)
        self.right.isp.link(self.right_manipRescale.inputImage)

        # Linking depth
        self.left_manipRescale.out.link(self.depth.left)
        self.right_manipRescale.out.link(self.depth.right)
        self.depth.depth.link(self.xout_depth.input)
        self.depth.disparity.link(self.xout_disparity.input)

        # Linking left
        self.depth.rectifiedLeft.link(self.xout_left.input)

        # Linking right
        self.depth.rectifiedRight.link(self.xout_right.input)

        return pipeline

    def create_pipeline(self) -> dai.Pipeline:
        pipeline = self.pipeline_basis()

        # Configuring depth node
        left_socket = get_socket_from_name("left", self.cam_config.name_to_socket)
        self.depth = pipeline.createStereoDepth()
        self.depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        self.depth.setLeftRightCheck(True)
        self.depth.setExtendedDisparity(False)
        self.depth.setSubpixel(True)
        self.depth.setDepthAlign(left_socket)
        self.depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

        pipeline = self.create_output_streams(pipeline)
        self.depth_max_disparity = self.depth.getMaxDisparity()
        return self.link_pipeline(pipeline)
