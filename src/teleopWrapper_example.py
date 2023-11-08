import subprocess as sp

from depthai_wrappers.teleop_wrapper import TeleopWrapper

w = TeleopWrapper(
    "/home/antoine/Pollen/pollen-vision/config_files/CONFIG_IMX296.json",
    60,
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
    data = w.get_data()
    for name, packets in data.items():
        procs[name].stdin.write(packets)
