from pathlib import Path

import depthai as dai
import numpy as np
from pollen_vision.camera_wrappers.depthai.calibration.undistort import (
    compute_undistort_maps,
    get_mesh,
)
from pollen_vision.camera_wrappers.depthai.cam_config import CamConfig
from pollen_vision.camera_wrappers.depthai.utils import get_config_file_path


def test_calib() -> None:
    path = str((Path(__file__).parent / "data" / Path("calibration_unit_tests.json")).resolve().absolute())
    path_res = str((Path(__file__).parent / "data" / Path("undistort_maps.npz")).resolve().absolute())

    c = CamConfig(get_config_file_path("CONFIG_OAK_D_PRO"), 60, resize=(1280, 720), exposure_params=None)
    c.calib = dai.CalibrationHandler(path)
    c.set_undistort_resolution((1280, 720))

    mapXL, mapYL, mapXR, mapYR = compute_undistort_maps(c)

    data = np.load(path_res)

    assert np.array_equal(mapXL, data["mapXL"])
    assert np.array_equal(mapYL, data["mapYL"])
    assert np.array_equal(mapXR, data["mapXR"])
    assert np.array_equal(mapYR, data["mapYR"])


def test_meshes() -> None:
    path_maps = str((Path(__file__).parent / "data" / Path("undistort_maps.npz")).resolve().absolute())
    path_meshes = str((Path(__file__).parent / "data" / Path("meshes.npz")).resolve().absolute())

    maps = np.load(path_maps)
    meshes = np.load(path_meshes)

    c = CamConfig(get_config_file_path("CONFIG_OAK_D_PRO"), 60, resize=(1280, 720), exposure_params=None)
    c.set_undistort_maps(maps["mapXL"], maps["mapYL"], maps["mapXR"], maps["mapYR"])

    mesh_left, meshWidth, meshHeight = get_mesh(c, "left")

    assert np.array_equal(mesh_left, meshes["mesh_left"])
    assert meshHeight == 46
    assert meshWidth == 80

    mesh_right, meshWidth, meshHeight = get_mesh(c, "right")

    assert np.array_equal(mesh_right, meshes["mesh_right"])
    assert meshHeight == 46
    assert meshWidth == 80
