import argparse
import subprocess as sp

from depthai_wrappers.teleop_wrapper import TeleopWrapper

argParser = argparse.ArgumentParser(description="teleop wrapper example")
argParser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to the configuration file.",
)
args = argParser.parse_args()

w = TeleopWrapper(
    args.config,
    50,
    rectify=True,
)


def spawn_procs(names: list[str]) -> dict:
    width, height = 1280, 720
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
    data, lat, _ = w.get_data()
    print(lat)
    for name, packets in data.items():
        procs[name].stdin.write(packets)
