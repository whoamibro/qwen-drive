import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import os
from pathlib import Path
from PIL import Image
import cv2


@dataclass
class CameraData:
    """Container for camera information"""
    image_path: str
    intrinsic: np.ndarray  # 3x3 intrinsic matrix
    sensor2ego_rotation: List[float]  # Quaternion [w, x, y, z] from sensor to ego
    sensor2ego_translation: List[float]  # Translation [x, y, z] from sensor to ego
    ego2sensor_rotation: Optional[List[float]] = None  # Quaternion [w, x, y, z] from ego to sensor
    ego2sensor_translation: Optional[List[float]] = None  # Translation [x, y, z] from ego to sensor
    ego2global_rotation: Optional[List[float]] = None  # Quaternion [w, x, y, z] from ego to global
    ego2global_translation: Optional[List[float]] = None  # Translation [x, y, z] from ego to global
    timestamp: int = 0
    image: Optional[np.ndarray] = None  # Loaded image data (H, W, C)


@dataclass
class NuScenesSample:
    """Container for a single nuScenes sample with all required data"""
    sample_idx: int
    token: str

    # 6 view cameras in order: FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_LEFT, BACK_RIGHT
    cameras: Dict[str, CameraData]

    # GT 3D bounding boxes: (N, 7) - [x, y, z, length, width, height, yaw]
    gt_boxes: np.ndarray

    # GT velocity: (N, 2) - [vx, vy] for each object
    gt_velocity: np.ndarray

    # GT planning trajectory: (1, 6, 3) - [x, y, z] for 6 waypoints
    gt_planning: np.ndarray

    # GT planning command: integer (e.g., 0-5 for different commands)
    gt_planning_command: int

    # GT navigation command: integer (0=turn_left, 1=turn_right, 2=go_straight, 3=follow_lane, 4=change_lane_to_left, 5=change_lane_to_right, 6=U-Turn)
    gt_navigation_command: Optional[int] = None

    # CAN bus data: (13,) array containing ego vehicle telemetry
    # Indices 6-8 contain ego velocity: [vx, vy, vz]
    can_bus: Optional[np.ndarray] = None

    # Additional useful information
    gt_names: Optional[List[str]] = None  # Object class names
    scene_token: Optional[str] = None
    timestamp: Optional[int] = None
    description: Optional[str] = None
    location: Optional[str] = None


def quaternion_inverse(q: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Compute the inverse of a quaternion.

    Args:
        q: Quaternion in [w, x, y, z] format

    Returns:
        Inverse quaternion as numpy array
    """
    q = np.array(q)
    # For unit quaternions, inverse is the conjugate
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_multiply(q1: Union[List[float], np.ndarray],
                       q2: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Multiply two quaternions: q1 * q2

    Args:
        q1: First quaternion in [w, x, y, z] format
        q2: Second quaternion in [w, x, y, z] format

    Returns:
        Result quaternion as numpy array
    """
    q1 = np.array(q1)
    q2 = np.array(q2)

    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quaternion_to_rotation_matrix(q: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Convert quaternion to 3x3 rotation matrix.

    Coordinate conventions:
    - Global: ENU (East-North-Up)
    - Ego vehicle: FLU (Forward-Left-Up)
    - Camera: RDF (Right-Down-Forward)

    Args:
        q: Quaternion in [w, x, y, z] format

    Returns:
        3x3 rotation matrix
    """
    q = np.array(q)
    w, x, y, z = q[0], q[1], q[2], q[3]

    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


def compute_ego2sensor_transform(sensor2ego_rotation: List[float],
                                 sensor2ego_translation: List[float]) -> Tuple[List[float], List[float]]:
    """
    Compute ego2sensor transformation from sensor2ego.

    Coordinate conventions:
    - Ego vehicle: FLU (Forward-Left-Up)
    - Camera: RDF (Right-Down-Forward)

    Args:
        sensor2ego_rotation: Quaternion [w, x, y, z] from sensor to ego
        sensor2ego_translation: Translation [x, y, z] from sensor to ego

    Returns:
        Tuple of (ego2sensor_rotation, ego2sensor_translation)
    """
    # Inverse rotation is quaternion conjugate for unit quaternions
    ego2sensor_rot = quaternion_inverse(sensor2ego_rotation).tolist()

    # Inverse translation: R^T * (-t)
    R = quaternion_to_rotation_matrix(sensor2ego_rotation)
    t = np.array(sensor2ego_translation)
    ego2sensor_trans = (-R.T @ t).tolist()

    return ego2sensor_rot, ego2sensor_trans


def compute_sensor2global_transform(sensor2ego_rotation: List[float],
                                    sensor2ego_translation: List[float],
                                    ego2global_rotation: List[float],
                                    ego2global_translation: List[float]) -> Tuple[List[float], List[float]]:
    """
    Compute sensor2global transformation from sensor2ego and ego2global.

    Coordinate conventions:
    - Global: ENU (East-North-Up)
    - Ego vehicle: FLU (Forward-Left-Up)
    - Camera: RDF (Right-Down-Forward)

    Args:
        sensor2ego_rotation: Quaternion [w, x, y, z] from sensor to ego
        sensor2ego_translation: Translation [x, y, z] from sensor to ego
        ego2global_rotation: Quaternion [w, x, y, z] from ego to global
        ego2global_translation: Translation [x, y, z] from ego to global

    Returns:
        Tuple of (sensor2global_rotation, sensor2global_translation)
    """
    # Rotation: sensor2global = ego2global * sensor2ego
    sensor2global_rot = quaternion_multiply(ego2global_rotation, sensor2ego_rotation).tolist()

    # Translation: sensor2global_trans = ego2global_R @ sensor2ego_trans + ego2global_trans
    ego2global_R = quaternion_to_rotation_matrix(ego2global_rotation)
    sensor2ego_trans = np.array(sensor2ego_translation)
    ego2global_trans = np.array(ego2global_translation)

    sensor2global_trans = (ego2global_R @ sensor2ego_trans + ego2global_trans).tolist()

    return sensor2global_rot, sensor2global_trans


def get_bbox_corners_3d(bbox: np.ndarray) -> np.ndarray:
    """
    Get the 8 corners of a 3D bounding box.

    Args:
        bbox: 3D bbox [x, y, z, length, width, height, yaw]

    Returns:
        corners: (8, 3) array of corner coordinates in ego frame
    """
    x, y, z, length, width, height, yaw = bbox

    # Create corners in bbox local frame (centered at origin)
    # In nuScenes, length is along x, width is along y, height is along z
    l, w, h = length / 2, width / 2, height / 2

    corners = np.array([
        [-l, -w, -h], [l, -w, -h], [l, w, -h], [-l, w, -h],  # bottom 4 corners
        [-l, -w, h], [l, -w, h], [l, w, h], [-l, w, h]       # top 4 corners
    ])

    # Rotation matrix around z-axis
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rot_mat = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])

    # Rotate corners
    corners = corners @ rot_mat.T

    # Translate to bbox center
    corners += np.array([x, y, z])

    return corners


def project_points_to_camera(points: np.ndarray,
                             camera_data: 'CameraData') -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points from ego frame to camera image coordinates.

    Args:
        points: (N, 3) array of 3D points in ego frame
        camera_data: CameraData object containing extrinsics and intrinsics

    Returns:
        image_points: (N, 2) array of 2D points in image coordinates
        depths: (N,) array of depths (z-coordinates in camera frame)
    """
    # Transform points from ego frame to camera frame
    ego2sensor_rot = quaternion_to_rotation_matrix(camera_data.ego2sensor_rotation)
    ego2sensor_trans = np.array(camera_data.ego2sensor_translation)

    # Apply transformation
    points_cam = points @ ego2sensor_rot.T + ego2sensor_trans

    # Extract depths (z in camera frame, which is forward direction in RDF)
    depths = points_cam[:, 2]

    # Project to image plane using intrinsic matrix
    # Convert to homogeneous coordinates
    points_2d = points_cam @ camera_data.intrinsic.T

    # Normalize by depth
    points_2d = points_2d[:, :2] / points_2d[:, 2:3]

    return points_2d, depths


def check_bbox_in_camera(bbox: np.ndarray,
                         camera_data: 'CameraData',
                         image_width: int = 1600,
                         image_height: int = 900,
                         min_visible_corners: int = 1) -> bool:
    """
    Check if a 3D bounding box is visible in a camera view.

    Args:
        bbox: 3D bbox [x, y, z, length, width, height, yaw]
        camera_data: CameraData object
        image_width: Image width in pixels
        image_height: Image height in pixels
        min_visible_corners: Minimum number of corners that must be visible

    Returns:
        True if bbox is visible in camera, False otherwise
    """
    # Get 8 corners of the bbox
    corners = get_bbox_corners_3d(bbox)

    # Project corners to camera
    image_points, depths = project_points_to_camera(corners, camera_data)

    # Check if corners are in front of camera (positive depth)
    in_front = depths > 0.1  # At least 10cm in front

    # Check if corners are within image bounds
    x_in_bounds = (image_points[:, 0] >= 0) & (image_points[:, 0] < image_width)
    y_in_bounds = (image_points[:, 1] >= 0) & (image_points[:, 1] < image_height)
    in_image = x_in_bounds & y_in_bounds

    # A corner is visible if it's in front and in image bounds
    visible = in_front & in_image

    # Check if enough corners are visible
    num_visible = np.sum(visible)

    return num_visible >= min_visible_corners


class NuScenesDataLoader:
    """
    Data loader for nuScenes dataset to prepare data for Qwen3-VL VQA tasks.

    Coordinate System Conventions:
    - Global: ENU (East-North-Up)
    - Ego vehicle: FLU (Forward-Left-Up)
    - Camera: RDF (Right-Down-Forward)

    Usage:
        loader = NuScenesDataLoader('/path/to/nuscenes2d_ego_temporal_infos_val.pkl')
        sample = loader.get_sample(0)

        # Access camera images
        front_cam_path = sample.cameras['CAM_FRONT'].image_path

        # Access GT data
        boxes = sample.gt_boxes
        velocities = sample.gt_velocity
        planning_traj = sample.gt_planning
        command = sample.gt_planning_command

        # Load temporal sequence (N consecutive frames from same scene)
        temporal_samples = loader.get_temporal_samples(start_idx=0, n_frames=5)

        # Load and resize images
        sample_with_images = loader.load_sample_images(sample, resize_factor=2)
    """

    # Camera order modified for ego-centric view: FL, F, FR, RL, R, RR
    # This order provides a more intuitive left-to-right spatial understanding
    CAMERA_NAMES = [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT'
    ]

    # Rear cameras that need horizontal flipping for ego-centric view
    REAR_CAMERAS = {'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'}

    # Planning command descriptions (commonly used in nuScenes)
    COMMAND_DESCRIPTIONS = {
        0: "turn left",
        1: "turn right",
        2: "go straight",
        3: "follow lane",
        4: "change lane to left",
        5: "change lane to right",
        6: "U-Turn"
    }

    def __init__(self, pkl_path: str, data_root: str = None):
        """
        Initialize the nuScenes data loader.

        Args:
            pkl_path: Path to the nuscenes2d_ego_temporal_infos_val.pkl file
            data_root: Root directory for nuScenes data. If None, will be inferred from pkl_path
        """
        self.pkl_path = pkl_path

        # Set data root
        if data_root is None:
            # Infer from pkl_path (assume pkl is in the nuscenes directory)
            self.data_root = str(Path(pkl_path).parent)
        else:
            self.data_root = data_root

        # Load the pkl file
        print(f"Loading nuScenes data from: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        self.metadata = data.get('metadata', {})
        self.infos = data['infos']

        print(f"Loaded {len(self.infos)} samples from nuScenes dataset")

    def __len__(self) -> int:
        """Return number of samples in the dataset"""
        return len(self.infos)

    def _resolve_image_path(self, relative_path: str) -> str:
        """
        Resolve the full image path.

        Args:
            relative_path: Relative path from pkl (e.g., './data/nuscenes/samples/CAM_FRONT/...')

        Returns:
            Full absolute path to the image
        """
        # Remove './' prefix if present
        if relative_path.startswith('./'):
            relative_path = relative_path[2:]

        # If path starts with 'data/nuscenes', extract the part after that
        if relative_path.startswith('data/nuscenes/'):
            # Extract path after 'data/nuscenes/'
            relative_to_nuscenes = relative_path[len('data/nuscenes/'):]
            full_path = os.path.join(self.data_root, relative_to_nuscenes)
        else:
            # Just join with data_root
            full_path = os.path.join(self.data_root, relative_path)

        return full_path

    def get_sample(self, idx: int) -> NuScenesSample:
        """
        Get a single sample by index.

        Args:
            idx: Sample index (0 to len(self)-1)

        Returns:
            NuScenesample object containing all the data
        """
        if idx < 0 or idx >= len(self.infos):
            raise IndexError(f"Sample index {idx} out of range [0, {len(self.infos)-1}]")

        sample_info = self.infos[idx]

        # Extract camera data
        cameras = {}
        for cam_name in self.CAMERA_NAMES:
            if cam_name not in sample_info['cams']:
                raise KeyError(f"Camera {cam_name} not found in sample {idx}")

            cam_info = sample_info['cams'][cam_name]

            # Compute ego2sensor transform
            ego2sensor_rot, ego2sensor_trans = compute_ego2sensor_transform(
                cam_info['sensor2ego_rotation'],
                cam_info['sensor2ego_translation']
            )

            cameras[cam_name] = CameraData(
                image_path=self._resolve_image_path(cam_info['data_path']),
                intrinsic=cam_info['cam_intrinsic'],
                sensor2ego_rotation=cam_info['sensor2ego_rotation'],
                sensor2ego_translation=cam_info['sensor2ego_translation'],
                ego2sensor_rotation=ego2sensor_rot,
                ego2sensor_translation=ego2sensor_trans,
                ego2global_rotation=cam_info.get('ego2global_rotation', None),
                ego2global_translation=cam_info.get('ego2global_translation', None),
                timestamp=cam_info['timestamp']
            )

        # Create the sample object
        sample = NuScenesSample(
            sample_idx=idx,
            token=sample_info['token'],
            cameras=cameras,
            gt_boxes=sample_info['gt_boxes'],
            gt_velocity=sample_info['gt_velocity'],
            gt_planning=sample_info['gt_planning'],
            gt_planning_command=sample_info['gt_planning_command'],
            gt_navigation_command=sample_info.get('gt_navigation_command', None),
            can_bus=sample_info.get('can_bus', None),
            gt_names=sample_info.get('gt_names', None),
            scene_token=sample_info.get('scene_token', None),
            timestamp=sample_info.get('timestamp', None),
            description=sample_info.get('description', None),
            location=sample_info.get('location', None)
        )

        return sample

    def get_samples_batch(self, indices: List[int]) -> List[NuScenesSample]:
        """
        Get multiple samples at once.

         Args:
            indices: List of sample indices

        Returns:
            List of NuScenesample objects
        """
        return [self.get_sample(idx) for idx in indices]

    def get_temporal_samples(self, start_idx: int, n_frames: int,
                            load_images: bool = False,
                            resize_factor: int = 1) -> Optional[List[NuScenesSample]]:
        """
        Load N consecutive temporal frames that belong to the same scene.

        This method ensures that all N frames have the same scene_token,
        which means they are from the same continuous driving sequence.

        Args:
            start_idx: Starting sample index
            n_frames: Number of consecutive frames to load
            load_images: Whether to load image data into memory
            resize_factor: Resize factor m (resize to 1/m of original size)

        Returns:
            List of N NuScenesSample objects if all frames are from same scene,
            None otherwise
        """
        if start_idx < 0 or start_idx + n_frames > len(self.infos):
            raise IndexError(
                f"Temporal sequence [{start_idx}, {start_idx + n_frames}) "
                f"out of range [0, {len(self.infos)}]"
            )

        # Get scene token from first frame
        scene_token = self.infos[start_idx]['scene_token']

        # Validate that all frames belong to the same scene
        for i in range(start_idx, start_idx + n_frames):
            if self.infos[i]['scene_token'] != scene_token:
                print(f"Warning: Frame {i} has different scene_token than frame {start_idx}")
                print(f"  Frame {start_idx} scene: {scene_token}")
                print(f"  Frame {i} scene: {self.infos[i]['scene_token']}")
                return None

        # Load all samples
        samples = []
        for i in range(start_idx, start_idx + n_frames):
            sample = self.get_sample(i)

            # Load images if requested
            if load_images:
                sample = self.load_sample_images(sample, resize_factor=resize_factor)

            samples.append(sample)

        return samples

    def load_image(self, image_path: str, resize_factor: int = 1,
                   flip_horizontal: bool = False) -> np.ndarray:
        """
        Load an image and optionally resize and/or flip it.

        Args:
            image_path: Path to the image file
            resize_factor: Resize factor m (resize to 1/m of original size)
            flip_horizontal: If True, flip the image horizontally (for ego-centric rear view)

        Returns:
            Image as numpy array (H, W, C) in RGB format
        """
        # Load image using PIL (handles various formats)
        img = Image.open(image_path).convert('RGB')

        # Resize if needed
        if resize_factor > 1:
            w, h = img.size
            new_w = w // resize_factor
            new_h = h // resize_factor
            img = img.resize((new_w, new_h), Image.LANCZOS)

        # Flip horizontally if needed (for rear cameras in ego-centric view)
        if flip_horizontal:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Convert to numpy array
        img_array = np.array(img)

        return img_array

    def load_sample_images(self, sample: NuScenesSample,
                          resize_factor: int = 1,
                          flip_rear_cameras: bool = True) -> NuScenesSample:
        """
        Load image data for all cameras in a sample.

        Args:
            sample: NuScenesSample object
            resize_factor: Resize factor m (resize to 1/m of original size)
            flip_rear_cameras: If True, flip rear camera images horizontally for ego-centric view

        Returns:
            Same sample with image data loaded into each CameraData object
        """
        for cam_name in self.CAMERA_NAMES:
            camera = sample.cameras[cam_name]
            # Flip rear cameras horizontally for ego-centric view
            flip = flip_rear_cameras and (cam_name in self.REAR_CAMERAS)
            camera.image = self.load_image(camera.image_path,
                                          resize_factor=resize_factor,
                                          flip_horizontal=flip)

        return sample

    def get_image_paths_for_sample(self, idx: int) -> List[str]:
        """
        Get image paths for all 6 cameras in standard order.

        Args:
            idx: Sample index

        Returns:
            List of 6 image paths in order: FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_LEFT, BACK_RIGHT
        """
        sample = self.get_sample(idx)
        return [sample.cameras[cam_name].image_path for cam_name in self.CAMERA_NAMES]

    def prepare_for_qwen3vl(self, idx: int) -> Dict:
        """
        Prepare a sample in a format suitable for Qwen3-VL VQA pipeline.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing all necessary data formatted for VQA
        """
        sample = self.get_sample(idx)

        # Get image paths in standard order
        image_paths = [sample.cameras[cam_name].image_path for cam_name in self.CAMERA_NAMES]

        # Extract camera parameters
        intrinsics = {cam_name: sample.cameras[cam_name].intrinsic
                     for cam_name in self.CAMERA_NAMES}

        extrinsics = {
            cam_name: {
                'sensor2ego_rotation': sample.cameras[cam_name].sensor2ego_rotation,
                'sensor2ego_translation': sample.cameras[cam_name].sensor2ego_translation,
                'ego2sensor_rotation': sample.cameras[cam_name].ego2sensor_rotation,
                'ego2sensor_translation': sample.cameras[cam_name].ego2sensor_translation
            }
            for cam_name in self.CAMERA_NAMES
        }

        # Get command description
        command_desc = self.COMMAND_DESCRIPTIONS.get(
            sample.gt_planning_command,
            f"unknown command ({sample.gt_planning_command})"
        )

        return {
            'sample_idx': idx,
            'token': sample.token,
            'scene_token': sample.scene_token,

            # Image paths in order
            'images': {
                'paths': image_paths,
                'cam_front': image_paths[0],
                'cam_front_left': image_paths[1],
                'cam_front_right': image_paths[2],
                'cam_back': image_paths[3],
                'cam_back_left': image_paths[4],
                'cam_back_right': image_paths[5],
            },

            # Camera parameters
            'camera_intrinsics': intrinsics,
            'camera_extrinsics': extrinsics,

            # Ground truth data
            'gt_boxes': sample.gt_boxes,
            'gt_velocity': sample.gt_velocity,
            'gt_planning_trajectory': sample.gt_planning,
            'gt_planning_command': sample.gt_planning_command,
            'gt_planning_command_desc': command_desc,

            # Additional info
            'gt_names': sample.gt_names,
            'num_objects': len(sample.gt_boxes),
            'description': sample.description,
            'location': sample.location,
            'timestamp': sample.timestamp,
        }

    def create_vqa_message(self, idx: int, question: str,
                          include_metadata: bool = True) -> List[Dict]:
        """
        Create a VQA message in the format expected by Qwen3-VL processor.

        Args:
            idx: Sample index
            question: Question text to ask about the scene
            include_metadata: Whether to include metadata in the question text

        Returns:
            Message list formatted for Qwen3-VL
        """
        data = self.prepare_for_qwen3vl(idx)

        # Build content list starting with images
        content = []

        # Add 6 view images
        for img_path in data['images']['paths']:
            content.append({"type": "image", "image": img_path})

        # Build question text
        question_text = question

        if include_metadata:
            metadata_text = f"\n\nScene Information:\n"
            metadata_text += f"- Location: {data['location']}\n"
            metadata_text += f"- Scene Description: {data['description']}\n"
            metadata_text += f"- Driving Command: {data['gt_planning_command_desc']}\n"
            metadata_text += f"- Number of Objects: {data['num_objects']}\n"

            question_text = question + metadata_text

        # Add text content
        content.append({"type": "text", "text": question_text})

        return [{"role": "user", "content": content}]

    def print_sample_summary(self, idx: int):
        """Print a summary of a sample for debugging."""
        sample = self.get_sample(idx)

        print(f"\n{'='*80}")
        print(f"Sample {idx} Summary")
        print(f"{'='*80}")
        print(f"Token: {sample.token}")
        print(f"Scene Token: {sample.scene_token}")
        print(f"Description: {sample.description}")
        print(f"Location: {sample.location}")
        print(f"Timestamp: {sample.timestamp}")
        print(f"\nPlanning Command: {sample.gt_planning_command} - "
              f"{self.COMMAND_DESCRIPTIONS.get(sample.gt_planning_command, 'unknown')}")
        print(f"\nGT Boxes Shape: {sample.gt_boxes.shape}")
        print(f"GT Velocity Shape: {sample.gt_velocity.shape}")
        print(f"GT Planning Shape: {sample.gt_planning.shape}")
        print(f"Number of Objects: {len(sample.gt_boxes)}")

        print(f"\nCamera Information:")
        for cam_name in self.CAMERA_NAMES:
            cam = sample.cameras[cam_name]
            print(f"  {cam_name}:")
            print(f"    Path: {cam.image_path}")
            print(f"    Exists: {os.path.exists(cam.image_path)}")
            print(f"    Intrinsic shape: {cam.intrinsic.shape}")
            print(f"    Sensor2Ego rotation (quaternion): {cam.sensor2ego_rotation}")
            print(f"    Sensor2Ego translation: {cam.sensor2ego_translation}")
            print(f"    Ego2Sensor rotation (quaternion): {cam.ego2sensor_rotation}")
            print(f"    Ego2Sensor translation: {cam.ego2sensor_translation}")
            if cam.image is not None:
                print(f"    Image loaded: shape {cam.image.shape}")

        if sample.gt_names is not None and len(sample.gt_names) > 0:
            print(f"\nObject Classes: {set(sample.gt_names)}")

        print(f"{'='*80}\n")


def main():
    """Example usage of the NuScenesDataLoader with all features"""

    # Initialize the loader
    pkl_path = os.environ.get("NUSCENES_PKL_PATH", "nuscenes2d_ego_temporal_infos_val.pkl")
    loader = NuScenesDataLoader(pkl_path)

    print(f"\nTotal samples: {len(loader)}")

    # ============================================================
    # Feature 1: Get a single sample (deterministic by index)
    # ============================================================
    print("\n" + "="*80)
    print("Feature 1: Get Single Sample by Index")
    print("="*80)
    sample_idx = 0
    sample = loader.get_sample(sample_idx)
    print(f"Sample {sample_idx} - Token: {sample.token}")
    print(f"Scene Token: {sample.scene_token}")
    print(f"Planning Command: {loader.COMMAND_DESCRIPTIONS[sample.gt_planning_command]}")

    # ============================================================
    # Feature 2: Load N temporal frames with scene validation
    # ============================================================
    print("\n" + "="*80)
    print("Feature 2: Load Temporal Sequence (N consecutive frames)")
    print("="*80)

    n_frames = 5
    temporal_samples = loader.get_temporal_samples(
        start_idx=0,
        n_frames=n_frames,
        load_images=False
    )

    if temporal_samples is not None:
        print(f"✓ Successfully loaded {len(temporal_samples)} consecutive frames from same scene")
        print(f"  Scene Token: {temporal_samples[0].scene_token}")
        for i, s in enumerate(temporal_samples):
            print(f"    Frame {i}: idx={s.sample_idx}, token={s.token[:8]}...")
    else:
        print(f"✗ Failed to load temporal sequence (different scene_tokens)")

    # ============================================================
    # Feature 3: Load images with resizing
    # ============================================================
    print("\n" + "="*80)
    print("Feature 3: Load and Resize Images")
    print("="*80)

    # Load without resizing
    sample_with_images = loader.load_sample_images(sample, resize_factor=1)
    front_cam = sample_with_images.cameras['CAM_FRONT']
    print(f"Original image shape: {front_cam.image.shape}")

    # Load with resizing (1/2 size)
    sample_resized = loader.get_sample(sample_idx)
    sample_resized = loader.load_sample_images(sample_resized, resize_factor=2)
    front_cam_resized = sample_resized.cameras['CAM_FRONT']
    print(f"Resized image shape (1/2): {front_cam_resized.image.shape}")

    # Load with resizing (1/4 size)
    sample_resized_4 = loader.get_sample(sample_idx)
    sample_resized_4 = loader.load_sample_images(sample_resized_4, resize_factor=4)
    front_cam_resized_4 = sample_resized_4.cameras['CAM_FRONT']
    print(f"Resized image shape (1/4): {front_cam_resized_4.image.shape}")

    # ============================================================
    # Feature 4: Ego2Sensor extrinsics computation
    # ============================================================
    print("\n" + "="*80)
    print("Feature 4: Ego2Sensor Extrinsics (FLU -> RDF)")
    print("="*80)

    for cam_name in ['CAM_FRONT', 'CAM_BACK']:
        cam = sample.cameras[cam_name]
        print(f"\n{cam_name}:")
        print(f"  Sensor2Ego rotation: {cam.sensor2ego_rotation}")
        print(f"  Sensor2Ego translation: {cam.sensor2ego_translation}")
        print(f"  Ego2Sensor rotation: {cam.ego2sensor_rotation}")
        print(f"  Ego2Sensor translation: {cam.ego2sensor_translation}")

    # ============================================================
    # Feature 5: Load temporal sequence WITH images
    # ============================================================
    print("\n" + "="*80)
    print("Feature 5: Load Temporal Sequence WITH Resized Images")
    print("="*80)

    n_frames_img = 3
    temporal_with_images = loader.get_temporal_samples(
        start_idx=0,
        n_frames=n_frames_img,
        load_images=True,
        resize_factor=4  # Resize to 1/4
    )

    if temporal_with_images is not None:
        print(f"✓ Loaded {len(temporal_with_images)} frames with resized images")
        for i, s in enumerate(temporal_with_images):
            img_shape = s.cameras['CAM_FRONT'].image.shape
            print(f"    Frame {i}: CAM_FRONT image shape = {img_shape}")

    # ============================================================
    # Prepare for Qwen3-VL
    # ============================================================
    print("\n" + "="*80)
    print("Prepare Data for Qwen3-VL")
    print("="*80)

    vqa_data = loader.prepare_for_qwen3vl(sample_idx)
    print("VQA Data Keys:", list(vqa_data.keys()))
    print(f"Number of objects: {vqa_data['num_objects']}")
    print(f"Planning command: {vqa_data['gt_planning_command_desc']}")

    # Create a VQA message
    question = """Analyze the driving scene from these 6 camera views.
    Describe the environment, identify critical objects, and assess
    the ego vehicle's driving behavior."""

    messages = loader.create_vqa_message(sample_idx, question)
    print(f"\nCreated VQA message with {len(messages[0]['content'])} content items")
    print(f"  - Images: {sum(1 for c in messages[0]['content'] if c['type'] == 'image')}")
    print(f"  - Text prompts: {sum(1 for c in messages[0]['content'] if c['type'] == 'text')}")

    print("\n" + "="*80)
    print("All features demonstrated successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
