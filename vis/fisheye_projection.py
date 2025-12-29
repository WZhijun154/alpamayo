#!/usr/bin/env python3
"""
Proper fisheye camera projection using actual calibration parameters.
Uses the polynomial distortion model from the camera intrinsics.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import physical_ai_av
from scipy.spatial.transform import Rotation
from typing import Tuple, Dict, List
import os


class FisheyeProjector:
    """Proper fisheye camera projection using polynomial distortion model."""

    def __init__(self, clip_id: str):
        """Initialize with actual camera calibration data."""
        self.clip_id = clip_id
        self.avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
        self._load_calibration()

    def _load_calibration(self):
        """Load camera intrinsics and extrinsics from dataset."""
        print("📷 Loading fisheye camera calibration data...")

        self.intrinsics = self.avdi.get_clip_feature(
            self.clip_id,
            self.avdi.features.CALIBRATION.CAMERA_INTRINSICS,
            maybe_stream=True
        )

        self.extrinsics = self.avdi.get_clip_feature(
            self.clip_id,
            self.avdi.features.CALIBRATION.SENSOR_EXTRINSICS,
            maybe_stream=True
        )

        # Filter for camera sensors only
        camera_names = [name for name in self.extrinsics.index if 'camera' in name]
        self.camera_extrinsics = self.extrinsics.loc[camera_names]

        print(f"✅ Loaded fisheye calibration for {len(camera_names)} cameras")

    def _get_camera_pose(self, camera_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get camera pose (rotation and translation) in vehicle frame."""
        if camera_name not in self.camera_extrinsics.index:
            raise ValueError(f"Camera {camera_name} not found in extrinsics data")

        extrinsics = self.camera_extrinsics.loc[camera_name]
        translation = np.array([extrinsics['x'], extrinsics['y'], extrinsics['z']])
        rotation = Rotation.from_quat([extrinsics['qx'], extrinsics['qy'], extrinsics['qz'], extrinsics['qw']])
        rotation_matrix = rotation.as_matrix()

        return rotation_matrix, translation

    def _fisheye_project(self, points_3d: np.ndarray, camera_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Project 3D points to fisheye image using polynomial distortion model.

        Args:
            points_3d: Array of shape [N, 3] in camera coordinate frame
            camera_name: Name of camera for intrinsic parameters

        Returns:
            Tuple of (projected_points, valid_mask)
        """
        if camera_name not in self.intrinsics.index:
            return np.full((len(points_3d), 2), np.nan), np.zeros(len(points_3d), dtype=bool)

        calib = self.intrinsics.loc[camera_name]

        # Extract calibration parameters
        cx = calib['cx']
        cy = calib['cy']
        fw_poly = [calib[f'fw_poly_{i}'] for i in range(5)]  # Forward polynomial coefficients

        # Only project points in front of camera (positive Z)
        valid_mask = points_3d[:, 2] > 0.1

        if valid_mask.sum() == 0:
            return np.full((len(points_3d), 2), np.nan), valid_mask

        valid_points = points_3d[valid_mask]

        # Convert to spherical coordinates
        x, y, z = valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]

        # Calculate radius in XY plane and total radius
        rho = np.sqrt(x**2 + y**2)
        r = np.sqrt(x**2 + y**2 + z**2)

        # Avoid division by zero
        safe_mask = (r > 1e-8) & (z > 1e-8)

        # Calculate incident angle (angle from optical axis)
        theta = np.zeros_like(rho)
        theta[safe_mask] = np.arccos(np.clip(z[safe_mask] / r[safe_mask], -1, 1))

        # Apply fisheye polynomial distortion model
        # Forward model: radius_image = poly(theta)
        r_image = np.zeros_like(theta)
        for i, coeff in enumerate(fw_poly):
            if i == 0:
                continue  # Skip constant term (usually 0)
            r_image += coeff * (theta ** i)

        # Convert back to image coordinates
        projected_points = np.full((len(points_3d), 2), np.nan)

        if len(valid_points) > 0:
            # Calculate image coordinates for valid points
            valid_r_image = r_image[safe_mask]
            valid_rho = rho[safe_mask]
            valid_x = x[safe_mask]
            valid_y = y[safe_mask]

            # Normalize by rho to get unit direction, then scale by fisheye radius
            scale_factor = np.zeros_like(valid_rho)
            nonzero_rho = valid_rho > 1e-8
            scale_factor[nonzero_rho] = valid_r_image[nonzero_rho] / valid_rho[nonzero_rho]

            # Image coordinates
            u = cx + scale_factor * valid_x
            v = cy + scale_factor * valid_y

            # Map back to full array
            full_safe_mask = np.zeros(len(points_3d), dtype=bool)
            full_safe_mask[valid_mask] = safe_mask

            projected_points[full_safe_mask, 0] = u
            projected_points[full_safe_mask, 1] = v

            # Update valid mask to include bounds checking
            width = calib['width']
            height = calib['height']

            in_bounds = (
                (projected_points[:, 0] >= 0) &
                (projected_points[:, 0] < width) &
                (projected_points[:, 1] >= 0) &
                (projected_points[:, 1] < height)
            )
            valid_mask = valid_mask & in_bounds

        return projected_points, valid_mask

    def project_trajectory_to_camera(self, trajectory_xyz: np.ndarray, camera_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Project 3D trajectory to fisheye camera image."""

        # Get camera pose
        R_cam, t_cam = self._get_camera_pose(camera_name)

        # Transform to camera coordinate frame
        # First translate, then rotate with inverse rotation (transpose)
        points_cam = (R_cam.T @ (trajectory_xyz - t_cam).T).T

        # Project using fisheye model
        projected, valid = self._fisheye_project(points_cam, camera_name)

        return projected, valid

    def get_available_cameras(self) -> List[str]:
        """Get list of available camera names."""
        return [name for name in self.intrinsics.index if 'camera' in name]


def test_fisheye_projection():
    """Test the fisheye projection with actual calibration."""

    print("🐟 Testing Fisheye Camera Projection")
    print("=" * 50)

    # Load test data
    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    projector = FisheyeProjector(clip_id)

    # Print calibration details
    for camera_name in ["camera_front_wide_120fov", "camera_front_tele_30fov"]:
        if camera_name in projector.intrinsics.index:
            calib = projector.intrinsics.loc[camera_name]
            print(f"\n📷 {camera_name}:")
            print(f"   Resolution: {calib['width']}x{calib['height']}")
            print(f"   Principal point: ({calib['cx']:.1f}, {calib['cy']:.1f})")
            print(f"   Forward poly: [{', '.join(f'{coeff:.3e}' for coeff in [calib[f'fw_poly_{i}'] for i in range(5)])}]")

    # Create test trajectory
    test_trajectory = np.array([
        [0, 0, 0],      # Start at vehicle origin
        [2, 0, 0],      # 2m forward
        [5, 0, 0],      # 5m forward
        [10, 1, 0],     # 10m forward, 1m left
        [15, 2, 0],     # 15m forward, 2m left
        [20, 3, 0],     # 20m forward, 3m left
    ])

    print(f"\n🎯 Test trajectory:")
    for i, point in enumerate(test_trajectory):
        print(f"   Point {i}: ({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f})")

    # Test projection on different cameras
    for camera_name in ["camera_front_wide_120fov", "camera_cross_left_120fov", "camera_front_tele_30fov"]:
        if camera_name in projector.get_available_cameras():
            projected, valid = projector.project_trajectory_to_camera(test_trajectory, camera_name)

            visible_count = valid.sum()
            print(f"\n📊 {camera_name}:")
            print(f"   Visible points: {visible_count}/{len(test_trajectory)}")

            if visible_count > 0:
                valid_points = projected[valid]
                print(f"   Image coordinates range:")
                print(f"     u: [{valid_points[:, 0].min():.0f}, {valid_points[:, 0].max():.0f}]")
                print(f"     v: [{valid_points[:, 1].min():.0f}, {valid_points[:, 1].max():.0f}]")

                print(f"   First few projected points:")
                for i, (point, is_valid) in enumerate(zip(projected, valid)):
                    if is_valid and i < 3:
                        print(f"     Point {i}: u={point[0]:.1f}, v={point[1]:.1f}")


if __name__ == "__main__":
    test_fisheye_projection()