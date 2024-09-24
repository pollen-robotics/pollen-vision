import argparse
import logging
import subprocess as sp
from typing import Dict, List

from pollen_vision.camera_wrappers.depthai.teleop import TeleopWrapper
from pollen_vision.camera_wrappers.depthai.utils import (
    get_config_file_path,
    get_config_files_names,
)

valid_configs = get_config_files_names()

argParser = argparse.ArgumentParser(description="teleop wrapper example")
argParser.add_argument(
    "--config",
    type=str,
    required=True,
    choices=valid_configs,
    help=f"Configutation file name : {valid_configs}",
)
args = argParser.parse_args()

logging.basicConfig(level=logging.DEBUG)

w = TeleopWrapper(get_config_file_path(args.config), 60, rectify=True)


def spawn_procs(names: List[str]) -> Dict[str, sp.Popen]:  # type: ignore [type-arg]
    width, height = 960, 720
    command = [
        "ffplay",
        "-i",
        "-",
        "-x",
        str(width),
        "-y",
        str(height),
        "-framerate",
        "60",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-framedrop",
        "-strict",
        "experimental",
    ]

    procs = {}
    try:
        for name in names:
            procs[name] = sp.Popen(command, stdin=sp.PIPE)  # Start the ffplay process
    except Exception:
        exit("Error: cannot run ffplay!\nTry running: sudo apt install ffmpeg")

    return procs


procs = spawn_procs(["left", "right"])

while True:
    data, lat, _ = w.get_data_h264()
    logging.info(lat)
    for name, packets in data.items():
        # if name == "left_raw" or name == "right_raw":
        #    continue
        io = procs[name].stdin
        if io is not None:
            io.write(packets)
        else:
            logging.error(f"io error with {procs[name]}")
