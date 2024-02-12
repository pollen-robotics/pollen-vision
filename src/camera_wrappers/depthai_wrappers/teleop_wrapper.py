from datetime import timedelta
from typing import Dict, Optional, Tuple

import depthai as dai
import numpy as np
import numpy.typing as npt

from camera_wrappers.depthai_wrappers.wrapper import Wrapper


class TeleopWrapper(Wrapper):  # type: ignore[misc]
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
            resize=(1280, 720),
            rectify=rectify,
            exposure_params=exposure_params,
            mx_id=mx_id,
            isp_scale=(2, 3),
        )

    def get_data(
        self,
    ) -> Tuple[Dict[str, npt.NDArray[np.uint8]], Dict[str, float], Dict[str, timedelta]]:
        data, latency, ts = super().get_data()

        for name, pkt in data.items():
            data[name] = pkt.getData()

        return data, latency, ts

    def _link_pipeline(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        self.left.isp.link(self.left_manip.inputImage)
        self.left_manip.out.link(self.left_encoder.input)
        self.right.isp.link(self.right_manip.inputImage)
        self.right_manip.out.link(self.right_encoder.input)

        self.left_encoder.bitstream.link(self.xout_left.input)
        self.right_encoder.bitstream.link(self.xout_right.input)

        return pipeline

    def _create_encoders(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        profile = dai.VideoEncoderProperties.Profile.H264_MAIN
        bitrate = 4000
        numBFrames = 0  # gstreamer recommends 0 B frames
        self.left_encoder = pipeline.create(dai.node.VideoEncoder)
        self.left_encoder.setDefaultProfilePreset(self.cam_config.fps, profile)
        self.left_encoder.setKeyframeFrequency(self.cam_config.fps)  # every 1s
        self.left_encoder.setNumBFrames(numBFrames)
        self.left_encoder.setBitrateKbps(bitrate)

        self.right_encoder = pipeline.create(dai.node.VideoEncoder)
        self.right_encoder.setDefaultProfilePreset(self.cam_config.fps, profile)
        self.right_encoder.setKeyframeFrequency(self.cam_config.fps)  # every 1s
        self.right_encoder.setNumBFrames(numBFrames)
        self.right_encoder.setBitrateKbps(bitrate)

        return pipeline

    def _create_pipeline(self) -> dai.Pipeline:
        pipeline = self._pipeline_basis()

        pipeline = self._create_encoders(pipeline)

        pipeline = self._create_output_streams(pipeline)

        return self._link_pipeline(pipeline)

    def _create_queues(self) -> Dict[str, dai.DataOutputQueue]:
        # config for video: https://docs.luxonis.com/projects/api/en/latest/components/device/#output-queue-maxsize-and-blocking
        queues: Dict[str, dai.DataOutputQueue] = {}
        for name in ["left", "right"]:
            queues[name] = self._device.getOutputQueue(name, maxSize=10, blocking=True)
        return queues
