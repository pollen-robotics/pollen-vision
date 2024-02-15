import asyncio
import fractions
import time

import av

FPS = 15
VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 1 / FPS
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
RESOLUTION = (1280, 800)


class Recorder:
    _start: float
    _timestamp: int

    def __init__(self, filename):
        self.output = av.open(filename, "w")
        self.stream = self.output.add_stream("h264", str(FPS))
        self.stream.height = RESOLUTION[1]
        self.stream.width = RESOLUTION[0]
        self.stream.bit_rate = 8500e3

    def start(self):
        self._start = time.time()

    async def next_timestamp(self):
        if hasattr(self, "_timestamp"):
            self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
            wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
            await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0
        return self._timestamp, VIDEO_TIME_BASE

    async def new_im(self, im):
        pts, time_base = await self.next_timestamp()
        frame = av.video.frame.VideoFrame.from_ndarray(im, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        packet = self.stream.encode(frame)
        self.output.mux(packet)

    def stop(self):
        packet = self.stream.encode(None)
        self.output.mux(packet)
        self.output.close()
