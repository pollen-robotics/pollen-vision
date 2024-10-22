from datetime import timedelta
from typing import Dict, Tuple

import cv2
import depthai as dai
import numpy as np
import numpy.typing as npt
from pollen_vision.camera_wrappers.depthai.utils import (
    colorizeDepth,
    get_config_file_path,
    get_socket_from_name,
)
from pollen_vision.camera_wrappers.depthai.wrapper import DepthaiWrapper


class TOFWrapper(DepthaiWrapper):  # type: ignore
    def __init__(
        self,
        cam_config_json: str,
        fps: int = 30,
        force_usb2: bool = False,
        mx_id: str = "",
        crop: bool = False,
        noalign: bool = False,
        rectify: bool = False,
        create_pointcloud: bool = False,
    ) -> None:
        """
        Using create_pointcloud mode, we can't undistort at the same time (not enough resources on the device)
        """
        self.noalign = noalign
        self.create_pointcloud = create_pointcloud
        super().__init__(
            cam_config_json,
            fps,
            force_usb2=force_usb2,
            resize=None,
            rectify=rectify,
            exposure_params=None,
            mx_id=mx_id,
            # isp_scale=(2, 3),
        )
        self.cvColorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
        self.cvColorMap[0] = [0, 0, 0]
        self.cam_config.undistort_resolution = (640, 480)
        self.crop = crop

    def crop_image(
        self, im: npt.NDArray[np.uint8], depth: npt.NDArray[np.float32]
    ) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
        # threshold depth
        thresholded_depth = np.where(depth > 0, 1, 0).astype(np.uint8)
        depth_mask_bounding_box = cv2.boundingRect(thresholded_depth)
        cropped_image = im[
            depth_mask_bounding_box[1] : depth_mask_bounding_box[1] + depth_mask_bounding_box[3],
            depth_mask_bounding_box[0] : depth_mask_bounding_box[0] + depth_mask_bounding_box[2],
        ]
        cropped_depth = depth[
            depth_mask_bounding_box[1] : depth_mask_bounding_box[1] + depth_mask_bounding_box[3],
            depth_mask_bounding_box[0] : depth_mask_bounding_box[0] + depth_mask_bounding_box[2],
        ]

        cropped_image = cv2.resize(cropped_image, (640, 480))
        cropped_depth = cv2.resize(cropped_depth, (640, 480))

        return cropped_image, cropped_depth

    def get_data(self) -> Tuple[Dict[str, npt.NDArray[np.uint8]], Dict[str, float], Dict[str, timedelta]]:
        data: Dict[str, npt.NDArray[np.uint8]] = {}
        latency: Dict[str, float] = {}
        ts: Dict[str, timedelta] = {}

        messageGroup = self.queues["sync_out"].get()

        left = messageGroup["left"].getCvFrame()
        right = messageGroup["right"].getCvFrame()
        depth = messageGroup["depth_aligned"].getFrame()
        tof_intensity = messageGroup["tof_intensity"].getCvFrame()
        if self.create_pointcloud:
            inPointCloud = messageGroup["pcl"]
            points = inPointCloud.getPoints().astype(np.float64)
            points[:, 0] = -points[:, 0]  # Invert X axis

        # Temporary, not ideal

        if self.crop:
            cropped_left, cropped_depth = self.crop_image(left, depth)
            data["left"] = cropped_left
            data["depth"] = cropped_depth  # type: ignore
        else:
            data["left"] = left
            data["depth"] = depth

        data["right"] = right
        data["tof_intensity"] = tof_intensity

        if self.create_pointcloud:
            data["pointcloud"] = points
        return data, latency, ts

    def get_K(self) -> npt.NDArray[np.float32]:
        return super().get_K(left=True)  # type: ignore

    def _create_pipeline(self) -> dai.Pipeline:
        pipeline = self._pipeline_basis(create_imagemanip=not self.create_pointcloud)

        self.cam_tof = pipeline.create(dai.node.Camera)
        self.cam_tof.setFps(self.cam_config.fps)
        self.tof_socket = get_socket_from_name("tof", self.cam_config.name_to_socket)
        self.cam_tof.setBoardSocket(self.tof_socket)

        self.tof = pipeline.create(dai.node.ToF)

        if self.create_pointcloud:
            self.pointcloud = pipeline.create(dai.node.PointCloud)

        # === Tof configuration ===
        self.tofConfig = self.tof.initialConfig.get()
        self.tofConfig.enableFPPNCorrection = True
        self.tofConfig.enableOpticalCorrection = True  # True is ok ?
        self.tofConfig.enablePhaseShuffleTemporalFilter = True
        self.tofConfig.phaseUnwrappingLevel = 1
        self.tofConfig.phaseUnwrapErrorThreshold = 300
        self.tofConfig.enableTemperatureCorrection = False  # Not yet supported
        self.tofConfig.enableWiggleCorrection = False
        self.tofConfig.median = dai.MedianFilter.KERNEL_7x7
        self.tofConfig.enableDistortionCorrection = True if self.cam_config.rectify else False
        self.tof.initialConfig.set(self.tofConfig)
        # ==========================

        self.align = pipeline.create(dai.node.ImageAlign)

        self.sync = pipeline.create(dai.node.Sync)
        self.sync.setSyncThreshold(timedelta(seconds=(0.5 / self.cam_config.fps)))

        pipeline = self._create_output_streams(pipeline)

        return self._link_pipeline(pipeline)

    def _create_output_streams(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        pipeline = super()._create_output_streams(pipeline)

        self.xout_sync = pipeline.create(dai.node.XLinkOut)
        self.xout_sync.setStreamName("sync_out")

        return pipeline

    def _link_pipeline(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        if not self.create_pointcloud:
            self.left.isp.link(self.left_manip.inputImage)
            self.right.isp.link(self.right_manip.inputImage)
        self.tof.intensity.link(self.sync.inputs["tof_intensity"])

        if not self.create_pointcloud:
            self.left_manip.out.link(self.sync.inputs["left"])
            self.right_manip.out.link(self.sync.inputs["right"])
        else:
            self.left.isp.link(self.sync.inputs["left"])
            self.right.isp.link(self.sync.inputs["right"])

        self.sync.inputs["left"].setBlocking(False)
        self.sync.inputs["right"].setBlocking(False)

        self.cam_tof.raw.link(self.tof.input)

        if self.noalign:
            self.tof.depth.link(self.sync.inputs["depth_aligned"])
        else:
            if not self.create_pointcloud:
                self.left_manip.out.link(self.align.inputAlignTo)
            else:
                self.left.isp.link(self.align.inputAlignTo)
            self.tof.depth.link(self.align.input)
            self.align.outputAligned.link(self.sync.inputs["depth_aligned"])

        if self.create_pointcloud:
            self.align.outputAligned.link(self.pointcloud.inputDepth)
            self.pointcloud.outputPointCloud.link(self.sync.inputs["pcl"])

        self.sync.out.link(self.xout_sync.input)

        return pipeline

    def _create_queues(self) -> Dict[str, dai.DataOutputQueue]:
        # queues: Dict[str, dai.DataOutputQueue] = super()._create_queues()

        queues: Dict[str, dai.DataOutputQueue] = {}
        queues["sync_out"] = self._device.getOutputQueue("sync_out", maxSize=8, blocking=False)

        return queues


if __name__ == "__main__":
    t = TOFWrapper(get_config_file_path("CONFIG_IMX296_TOF"), fps=30, crop=False, rectify=False, create_pointcloud=True)

    print(dai.__version__)
    while True:
        data, _, _ = t.get_data()  # type: ignore
        left = data["left"]

        depth = data["depth"]
        tof_intensity = data["tof_intensity"]

        cv2.imshow("left", left)
        cv2.imshow("tof intensity", tof_intensity)
        colorized_depth = colorizeDepth(data["depth"])
        blended = cv2.addWeighted(left, 0.5, colorized_depth, 0.5, 0)
        cv2.imshow("blended", blended)
        cv2.imshow("colorized_depth", colorized_depth)
        cv2.waitKey(1)
