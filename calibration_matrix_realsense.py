import numpy as np
import pyrealsense2 as rs

def recover_matrix_calib() -> np.ndarray:
    """
    Recover the calibration matrix from the depth stream.

    Returns:
    - np.ndarray: Calibration matrix.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 0, 0, rs.format.z16, 30)  # Configure depth stream
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

if __name__ == '__main__':
    calibration_matrix = recover_matrix_calib()
    print(calibration_matrix)
