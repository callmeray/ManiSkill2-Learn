"""
    Real RealSense Depth Sensor utilities
"""
import time
import numpy as np
import open3d as o3d
from copy import deepcopy
import cv2
import open3d as o3d
import pyrealsense2 as rs


def generate_point_cloud(depth_img, camera_intrinsic, camera_extrinsic) -> np.ndarray:

    img_width = depth_img.shape[0]
    img_height = depth_img.shape[1]
    fx = camera_intrinsic[0, 0]
    fy = camera_intrinsic[1, 1]
    cx = camera_intrinsic[0, 2]
    cy = camera_intrinsic[1, 2]

    cam_mat = o3d.camera.PinholeCameraIntrinsic(img_width, img_height, fx, fy, cx, cy)
    o3d_depth = o3d.geometry.Image(depth_img)
    o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(
        o3d_depth, cam_mat, camera_extrinsic
    )
    pcd = np.asarray(o3d_cloud.points)
    return pcd


class RealSense(object):
    def __init__(self, SN, camera_extrinsic_matrix=np.eye(4)):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(SN)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.cfg = self.pipeline.start(config)
        time.sleep(2)

        self.camera_extrinsic_matrix = camera_extrinsic_matrix.copy()

    def get_depth_and_pointcloud(self):
        for _ in range(1000):
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data()).copy()
            depth_profile = self.cfg.get_stream(rs.stream.depth)
            intr = depth_profile.as_video_stream_profile().get_intrinsics()
            camera_intrinsic_matrix = np.asarray(
                [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]]
            )
            depth_image = depth_image.astype(np.float32) * 0.001
            scene_pcd = generate_point_cloud(
                depth_image, camera_intrinsic_matrix, self.camera_extrinsic_matrix
            )

            return depth_image, scene_pcd

        raise ValueError("Time out")

    def get_rgb_image(self):
        for _ in range(1000):
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            return color_image

        raise ValueError("Time out")
