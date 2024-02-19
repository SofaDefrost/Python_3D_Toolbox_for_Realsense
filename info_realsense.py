import numpy as np
import pyrealsense2 as rs

from typing import List


def get_matrix_calib(width: int, height: int, serial_number: str = "") -> np.ndarray:
    """
    Recover the calibration matrix from a RealSense camera.

    Args:
        width (int): Width of the depth stream.
        height (int): Height of the depth stream.
        serial_number (str, optional): Serial number of the device. Serial number of the device. Defaults to "" : it means that it will choose the camera automatically (useful when only one camera is connected).

    Returns:
        np.ndarray: Calibration matrix.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    if len(serial_number) > 0:
        config.enable_device(serial_number)
    config.enable_stream(rs.stream.depth, width, height,
                         rs.format.z16, 30)  # Configure depth stream
    profile = pipeline.start(config)

    depth_profile = profile.get_stream(rs.stream.depth)
    depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

    # Get parameters of the calibration matrix
    fx, fy, cx, cy = depth_intrinsics.fx, depth_intrinsics.fy, depth_intrinsics.ppx, depth_intrinsics.ppy

    # Calibration matrix
    calibration_matrix = np.array([[fx, 0, cx],
                                   [0, fy, cy],
                                   [0, 0, 1]], dtype=np.float32)
    return calibration_matrix


def get_serial_number() -> List[str]:
    """
    Recover the serial numbers of connected RealSense cameras.

    Returns:
        list[str]: List of serial numbers of connected cameras.
    """
    detected_camera = []
    realsense_ctx = rs.context()
    for i in range(len(realsense_ctx.devices)):
        detected_camera.append(realsense_ctx.devices[i].get_info(
            rs.camera_info.serial_number))
    return detected_camera


if __name__ == '__main__':
    print("Serial number(s) of camera(s) connected")
    serial_number = recover_serial_number()
    print(serial_number)
    print("Calibration matrix (640,480)")
    calibration_matrix = recover_matrix_calib(640, 480, serial_number[0])
    print(calibration_matrix)
    print("Calibration matrix (1280,720)")
    calibration_matrix = recover_matrix_calib(1280, 720)
    print(calibration_matrix)
