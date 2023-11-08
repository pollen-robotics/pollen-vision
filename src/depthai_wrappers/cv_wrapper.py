from typing import Dict, Optional, Tuple

import depthai as dai
import numpy as np

from depthai_wrappers.wrapper import Wrapper


class CvWrapper(Wrapper):
    def __init__(
        self,
        cam_config_json: str,
        fps: int,
        force_usb2: bool = False,
        resize: Optional[Tuple[int, int]] = None,
        rectify: bool = False,
    ) -> None:
        super().__init__(
            cam_config_json, fps, force_usb2=force_usb2, resize=resize, rectify=rectify
        )

    def get_data(self) -> Dict[str, np.ndarray]:
        data: Dict[str, np.ndarray] = {}
        for name, queue in self.queues.items():
            data[name] = queue.get().getCvFrame()
        return data

    def link_pipeline(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        # Linking left
        self.left.isp.link(self.left_manipRectify.inputImage)
        self.left_manipRectify.out.link(self.left_manipRescale.inputImage)
        self.left_manipRescale.out.link(self.xout_left.input)

        # Linking right
        self.right.isp.link(self.right_manipRectify.inputImage)
        self.right_manipRectify.out.link(self.right_manipRescale.inputImage)
        self.right_manipRescale.out.link(self.xout_right.input)

        return pipeline

    def create_pipeline(self) -> dai.Pipeline:
        pipeline = self.pipeline_basis()

        pipeline = self.create_output_streams(pipeline)

        return self.link_pipeline(pipeline)
