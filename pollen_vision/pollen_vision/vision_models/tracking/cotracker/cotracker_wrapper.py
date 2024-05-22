from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch


class CotrackerWrapper:
    def __init__(self, initial_points: List[List[np.uint8]] = []) -> None:
        self._device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        self._i = 0
        self._model = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online").to(
            self._device
        )  # ignore[no-untyped-call]
        self._is_first_step = True
        self._tracks = None
        self._points = initial_points
        self._window_size = self._model.step * 2
        self._frames_buffer: List[npt.NDArray[np.uint8]] = []

    def set_points(self, points: List[List[np.uint8]]) -> None:
        self._points = points

    def reset(self) -> None:
        self._is_first_step = True
        self._tracks = None
        self._frames_buffer = []
        self._i = 0

    def step(self, frame: npt.NDArray[np.uint8]) -> Tuple[Optional[npt.NDArray[np.float32]], Optional[npt.NDArray[np.float32]]]:
        self._frames_buffer.append(frame)
        if len(self._frames_buffer) < self._window_size:
            return None, None

        self._frames_buffer = self._frames_buffer[-self._window_size :]  # keep only the last window_size frames

        video_chunk = (
            torch.tensor(np.stack(self._frames_buffer), device=self._device)
            .float()
            .permute(0, 3, 1, 2)[None]  # ignore[arg-type]
        )  # (1, T, 3, H, W)

        # Converting the points to queries
        queries = np.zeros((1, len(self._points), 3))
        for point in self._points:
            queries[0, self._points.index(point), :] = (self._i, point[0], point[1])
        queries = torch.Tensor(queries).to(self._device)  # ignore[call-overload]

        pred_tracks, pred_visibility = self._model(video_chunk, is_first_step=self._is_first_step, grid_size=0, queries=queries)
        self._is_first_step = False

        self._i += 1
        if pred_tracks is None:
            return None, None

        return pred_tracks.cpu().numpy(), pred_visibility.cpu().numpy()


# if __name__ == "__main__":
#     point = None

#     def on_mouse(event, x: int, y: int, flags: Any, param: Any) -> None:
#         global point
#         if event == cv2.EVENT_LBUTTONDOWN:
#             point = (x, y)

#     cv2.namedWindow("frame")
#     cv2.setMouseCallback("frame", on_mouse)  # type: ignore

#     cotracker = CotrackerWrapper()
#     print("cotracker created")
#     vid = cv2.VideoCapture(0)
#     cotracker_initialized = False
#     while True:
#         ret, frame = vid.read()
#         if point is not None:
#             cotracker.reset()
#             cotracker.set_points([point])
#             point = None
#             cotracker_initialized = True

#         if cotracker_initialized:
#             tracks, visibility = cotracker.step(frame)
#             if tracks is not None:
#                 last_track = tracks[0, -1][0]
#                 frame = cv2.circle(frame, (int(last_track[0]), int(last_track[1])), 30, (0, 0, 255), -1)

#         # for point in points:
#         #     frame = cv2.circle(frame, point, 5, (0, 0, 255), -1)

#         cv2.imshow("frame", frame)
#         cv2.waitKey(1)
