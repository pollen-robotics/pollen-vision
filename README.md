# Installation

Install in "production" mode:
```console
pip install -e .
```

Install in "dev" mode:
```console
pip install -e .[dev]
```

# Run examples

There are three example scripts in `src/depthai_wrappers/`:
- cvWrapper_example.py -> returns raw cv2 images 
- teleopWrapper_example.py -> returns h264 packets encoded on board
- depthWrapper_example.py -> returns depth and disparity maps

# Calibration 

Check the [calibration](src/calibration/README.md) page for more details.
