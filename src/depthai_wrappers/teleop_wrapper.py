from typing import Tuple

import depthai as dai

from depthai_wrappers.wrapper import Wrapper


class TeleopWrapper(Wrapper):
    def __init__(
        self,
        cam_config_json: str,
        fps: int,
        force_usb2: bool = False,
        rectify: bool = False,
        exposure_params: Tuple[int, int] = None,
    ) -> None:
        super().__init__(
            cam_config_json,
            fps,
            force_usb2=force_usb2,
            resize=(1280, 720),
            rectify=rectify,
            exposure_params=exposure_params,
        )

    def get_data(self) -> Tuple:
        data, latency, ts = super().get_data()

        for name, pkt in data.items():
            data[name] = pkt.getData()

        return data, latency, ts

    def link_pipeline(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        self.left.isp.link(self.left_manipRectify.inputImage)
        # self.left_manipRectify.out.link(self.left_manipRescale.inputImage)
        self.right.isp.link(self.right_manipRectify.inputImage)
        # self.right_manipRectify.out.link(self.right_manipRescale.inputImage)

        self.left_manipRectify.out.link(self.left_encoder.input)
        self.right_manipRectify.out.link(self.right_encoder.input)

        self.left_encoder.bitstream.link(self.xout_left.input)
        self.right_encoder.bitstream.link(self.xout_right.input)

        return pipeline

    def create_encoders(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        profile = dai.VideoEncoderProperties.Profile.H264_MAIN
        self.left_encoder = pipeline.create(dai.node.VideoEncoder)
        self.left_encoder.setDefaultProfilePreset(self.cam_config.fps, profile)
        self.left_encoder.setKeyframeFrequency(self.cam_config.fps)  # every 1s
        self.left_encoder.setNumBFrames(0)  # gstreamer recommends 0 B frames
        self.left_encoder.setBitrateKbps(4000)

        self.right_encoder = pipeline.create(dai.node.VideoEncoder)
        self.right_encoder.setDefaultProfilePreset(self.cam_config.fps, profile)
        self.right_encoder.setKeyframeFrequency(self.cam_config.fps)  # every 1s
        self.right_encoder.setNumBFrames(0)  # gstreamer recommends 0 B frames
        self.right_encoder.setBitrateKbps(4000)

        return pipeline

    def create_manipResize(
        self, pipeline: dai.Pipeline, resolution: Tuple[int, int]
    ) -> dai.node.ImageManip:
        manipResize: dai.node.ImageManip = super().create_manipResize(
            pipeline, resolution
        )
        manipResize.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)

        return manipResize

    def create_manipRectify(
        self,
        pipeline: dai.Pipeline,
        cam_name: str,
        resolution: Tuple[int, int],
        rectify: bool = True,
    ) -> dai.node.ImageManip:
        manipRectify: dai.node.ImageManip = super().create_manipRectify(
            pipeline, cam_name, resolution, rectify
        )

        manipRectify.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)
        return manipRectify

    def create_pipeline(self) -> dai.Pipeline:
        pipeline = self.pipeline_basis()

        pipeline = self.create_encoders(pipeline)

        pipeline = self.create_output_streams(pipeline)

        return self.link_pipeline(pipeline)
