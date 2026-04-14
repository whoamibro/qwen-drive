#!/usr/bin/env python3
"""
Question-Answer Dataset Generator for nuScenes VQA

This module generates sample-specific question lists by:
1. Extracting prior knowledge from nuScenes .pkl data
2. Loading risk information from existing VQA results files
3. Building category-specific prompts for Qwen3-VL
4. Validating question templates by grounding placeholders against scene data
5. Collecting ALL applicable templates (not a fixed number)

Template Validation Process:
- For each template, inspect placeholder tags and their possible values
- VLM verifies which placeholder values can be grounded in 6-view images + prior
- A template is APPLICABLE if at least one semantically consistent combination
  of placeholder values can be grounded in the provided images/prior
- Templates are processed in batches (default 5) to avoid GPU OOM

Usage:
    # Show prompts only (no inference)
    python qa_generator.py --sample_idx 0 --category Observation --show_prompt

    # Run inference for a single category
    python qa_generator.py --sample_idx 0 --category Observation --run_inference

    # Run inference for all categories
    python qa_generator.py --sample_idx 0 --category all --run_inference

    # Custom batch size and model path
    python qa_generator.py --sample_idx 0 --category all --run_inference \\
        --batch_size 3 --model_path ./ckpts/qwen3_vl_235b_a22b_instruct

    # With custom distance filter (e.g., 30 meters)
    python qa_generator.py --sample_idx 0 --category all --run_inference --filter_distance 30

    # With separate rear filter for non-vehicle objects (e.g., 10 meters behind ego)
    python qa_generator.py --sample_idx 0 --category all --run_inference \\
        --filter_distance 20 --rear_filter 10

Output Files:
    - sample_X_qa_summary.json: Scene info and category statistics
    - sample_X_applicable_questions.json: All applicable templates with verified placeholders
    - sample_X_inference_detailed.json: Full debug info including raw VLM responses
"""

import os
import sys
import json
import glob
import argparse
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

# Optional imports for inference (only needed when --run_inference is used)
try:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False

from nuscenes_pipeline.core.nuscenes_data_loader import NuScenesDataLoader
from nuscenes_pipeline.core.nuscenes_prompt_generator import (
    check_bbox_in_camera,
    transform_velocity_to_global,
    transform_bbox_to_global,
    compute_ttc_obb_global
)

# ============================================================================
# CONSTANTS
# ============================================================================

VALID_CATEGORIES = [
    "Observation",
    "Identification",
    "Attributes_and_States",
    "Spatial_Relationships_and_Occlusion",
    "Traffic_Signs_and_Signals",
    "Road_Markings_and_Lane_Configuration",
    "Dynamic_Agents_and_Risk_Assessment",
    "Right_of_Way_and_Planning",
    "Environmental_and_Sensor_Conditions",
    "Causal_and_Hypothetical_Reasoning"
]

CAMERA_NAME_MAP = {
    'CAM_FRONT': 'Front',
    'CAM_FRONT_LEFT': 'Front-left',
    'CAM_FRONT_RIGHT': 'Front-right',
    'CAM_BACK': 'Rear',
    'CAM_BACK_LEFT': 'Rear-left',
    'CAM_BACK_RIGHT': 'Rear-right'
}

CAMERA_NAMES = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

# Rear cameras (for horizontal flip in egocentric mode)
REAR_CAMERAS = {'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'}

# Image labels matching risk assessment pipeline
CAM_LABELS = {
    "CAM_FRONT_LEFT": "Image 1: Front-left Camera",
    "CAM_FRONT": "Image 2: Front Camera",
    "CAM_FRONT_RIGHT": "Image 3: Front-right Camera",
    "CAM_BACK_LEFT": "Image 4: Rear-left Camera",
    "CAM_BACK": "Image 5: Rear Camera",
    "CAM_BACK_RIGHT": "Image 6: Rear-right Camera",
}

# Vehicle types - these are NOT filtered by rear_filter_distance
VEHICLE_TYPES = {'car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'motorcycle', 'bicycle'}

# Image number mapping (1-indexed, egocentric order)
CAM_TO_IMAGE_NUM = {
    'CAM_FRONT_LEFT': 1, 'CAM_FRONT': 2, 'CAM_FRONT_RIGHT': 3,
    'CAM_BACK_LEFT': 4, 'CAM_BACK': 5, 'CAM_BACK_RIGHT': 6
}


def _format_camera_refs(visible_cameras: List[str]) -> str:
    """Format visible cameras as 'Image N (DisplayName)' matching risk assessment format."""
    parts = []
    for cam_name in visible_cameras:
        img_num = CAM_TO_IMAGE_NUM[cam_name]
        display = CAMERA_NAME_MAP[cam_name]
        parts.append(f"Image {img_num} ({display})")
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts)

# ============================================================================
# SCENE ANALYSIS MODULE
# ============================================================================

class SceneAnalyzer:
    """Extract prior knowledge from nuScenes sample data."""

    def __init__(self, loader: NuScenesDataLoader):
        self.loader = loader

    def analyze_sample(self, sample_idx: int, max_distance: float = 20.0,
                       rear_filter_distance: float = None) -> Dict:
        """
        Analyze a sample and extract all prior knowledge.

        Args:
            sample_idx: Index of the sample to analyze
            max_distance: Maximum distance (in meters) from ego vehicle to include objects. Default 20.0m
            rear_filter_distance: Maximum distance for non-vehicle objects behind ego.
                                  If None, uses max_distance for all objects.

        Returns:
            Dictionary containing all extracted prior knowledge.
        """
        sample = self.loader.get_sample(sample_idx)

        # Extract ego vehicle information
        ego_info = self._extract_ego_info(sample)

        # Extract 3D object information with distance filtering
        objects_info = self._extract_objects_info(sample, max_distance, rear_filter_distance)

        # Extract object counts by class
        class_counts = self._count_objects_by_class(objects_info)

        # Extract objects by camera view
        objects_by_camera = self._group_objects_by_camera(objects_info)

        # Extract scene metadata
        scene_meta = {
            'sample_idx': sample_idx,
            'token': sample.token,
            'scene_token': sample.scene_token,
            'location': sample.location,
            'description': sample.description,
            'timestamp': sample.timestamp
        }

        return {
            'scene_meta': scene_meta,
            'ego_info': ego_info,
            'objects': objects_info,
            'class_counts': class_counts,
            'objects_by_camera': objects_by_camera,
            'total_objects': len(objects_info)
        }

    def _extract_ego_info(self, sample) -> Dict:
        """Extract ego vehicle information."""
        first_cam = sample.cameras[CAMERA_NAMES[0]]

        # Position and rotation
        ego_pos = first_cam.ego2global_translation
        ego_quat = first_cam.ego2global_rotation

        # Compute yaw
        qw, qx, qy, qz = ego_quat[0], ego_quat[1], ego_quat[2], ego_quat[3]
        ego_yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

        # Compute velocity (from next frame, with scene boundary check)
        ego_vel = np.array([0.0, 0.0])
        speed = 0.0
        try:
            next_sample = self.loader.get_sample(sample.sample_idx + 1)
            if next_sample.scene_token == sample.scene_token:
                next_cam = next_sample.cameras[CAMERA_NAMES[0]]
                pos_current = np.array(first_cam.ego2global_translation[:2])
                pos_next = np.array(next_cam.ego2global_translation[:2])
                dt = (next_sample.timestamp - sample.timestamp) / 1e6
                if dt > 0:
                    ego_vel = (pos_next - pos_current) / dt
                    speed = np.linalg.norm(ego_vel)
        except:
            pass

        # Driving command (use gt_navigation_command if available, fallback to gt_planning_command)
        COMMAND_MAP = {0: "Turn left", 1: "Turn right", 2: "Go straight", 3: "Follow lane", 4: "Change lane to left", 5: "Change lane to right", 6: "U-Turn"}
        cmd_idx = sample.gt_navigation_command if hasattr(sample, 'gt_navigation_command') and sample.gt_navigation_command is not None else sample.gt_planning_command
        driving_command = COMMAND_MAP.get(cmd_idx, "Unknown")

        # Determine motion state
        if speed < 0.1:
            motion_state = "stationary"
        elif speed < 5.0:
            motion_state = "slow"
        elif speed < 15.0:
            motion_state = "moderate"
        else:
            motion_state = "fast"

        return {
            'position': list(ego_pos),
            'yaw': float(ego_yaw),
            'yaw_degrees': float(np.degrees(ego_yaw)),
            'velocity': ego_vel.tolist(),
            'speed': float(speed),
            'speed_kmh': float(speed * 3.6),
            'motion_state': motion_state,
            'driving_command': driving_command,
            'quaternion': list(ego_quat)
        }

    def _extract_objects_info(self, sample, max_distance: float,
                               rear_filter_distance: float = None) -> List[Dict]:
        """
        Extract 3D object information with camera visibility and distance filtering.

        Args:
            sample: NuScenesSample object
            max_distance: Maximum distance (in meters) from ego vehicle to include objects.
            rear_filter_distance: Maximum distance for non-vehicle objects behind ego.
                                  If None, uses max_distance for all objects.

        Returns:
            List of object dictionaries with position, velocity, TTC, and visibility info.
        """
        objects = []

        if len(sample.gt_boxes) == 0:
            return objects

        # Get ego transformation
        first_cam = sample.cameras[CAMERA_NAMES[0]]
        ego2global_rot = first_cam.ego2global_rotation
        ego2global_trans = first_cam.ego2global_translation

        # Ego info for TTC
        ego_pos = np.array(ego2global_trans[:2])
        qw, qx, qy, qz = ego2global_rot
        ego_yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

        # Compute ego velocity (with scene boundary check)
        ego_vel = np.array([0.0, 0.0])
        try:
            next_sample = self.loader.get_sample(sample.sample_idx + 1)
            if next_sample.scene_token == sample.scene_token:
                next_cam = next_sample.cameras[CAMERA_NAMES[0]]
                pos_current = np.array(first_cam.ego2global_translation[:2])
                pos_next = np.array(next_cam.ego2global_translation[:2])
                dt = (next_sample.timestamp - sample.timestamp) / 1e6
                if dt > 0:
                    ego_vel = (pos_next - pos_current) / dt
        except:
            pass

        # Compute ego acceleration (centered difference, for acceleration-aware TTC)
        from nuscenes_pipeline.core.nuscenes_prompt_generator import compute_ego_acceleration
        ego_accel = compute_ego_acceleration(sample, self.loader)

        for obj_idx in range(len(sample.gt_boxes)):
            bbox = sample.gt_boxes[obj_idx]
            velocity = sample.gt_velocity[obj_idx]
            obj_name = sample.gt_names[obj_idx] if sample.gt_names is not None else "object"

            # Get object position in ego frame (FLU: Forward-Left-Up)
            obj_x = bbox[0]  # Forward/behind (positive = ahead)
            obj_y = bbox[1]  # Left/right (positive = left)

            # Calculate distance
            distance = np.sqrt(obj_x**2 + obj_y**2)

            # Filter by max distance
            if distance > max_distance:
                continue

            # Apply rear filter for non-vehicle objects (x < 0 means behind ego in FLU frame)
            if rear_filter_distance is not None and obj_x < 0:
                # Check if this is a non-vehicle object
                if obj_name.lower() not in VEHICLE_TYPES:
                    # Filter by rear_filter_distance for non-vehicles behind ego
                    if distance > rear_filter_distance:
                        continue

            # Find visible cameras
            visible_cameras = []
            for cam_name in CAMERA_NAMES:
                camera_data = sample.cameras[cam_name]
                if check_bbox_in_camera(bbox, camera_data):
                    visible_cameras.append(cam_name)

            if not visible_cameras:
                continue

            # Transform to global coordinates
            bbox_global = transform_bbox_to_global(bbox, ego2global_rot, ego2global_trans)
            vel_global = transform_velocity_to_global(velocity, ego2global_rot)

            # Compute TTC
            p_obj = np.array([bbox_global[0], bbox_global[1]])
            v_obj = np.array([vel_global[0], vel_global[1]])

            ttc_result = compute_ttc_obb_global(
                p_ego_ref=ego_pos,
                v_ego_global=ego_vel,
                yaw_ego=ego_yaw,
                p_obj_global=p_obj,
                v_obj_global=v_obj,
                yaw_obj_global=bbox_global[6],
                L_obj=bbox_global[3],
                W_obj=bbox_global[4],
                a_ego_global=ego_accel,
            )

            # Determine relative position (using obj_x, obj_y extracted earlier)
            if obj_x > 2:
                longitudinal = "ahead"
            elif obj_x < -2:
                longitudinal = "behind"
            else:
                longitudinal = "beside"

            if obj_y > 1:
                lateral = "left"
            elif obj_y < -1:
                lateral = "right"
            else:
                lateral = "center"

            # Determine motion state
            speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
            if speed < 0.1:
                obj_motion = "stationary"
            else:
                obj_motion = "moving"

            objects.append({
                'index': obj_idx,
                'category': obj_name,
                'distance': float(distance),
                'position_ego': [float(bbox[0]), float(bbox[1]), float(bbox[2])],
                'position_global': [float(bbox_global[0]), float(bbox_global[1]), float(bbox_global[2])],
                'dimensions': [float(bbox[3]), float(bbox[4]), float(bbox[5])],  # l, w, h
                'yaw': float(bbox[6]),
                'yaw_global': float(bbox_global[6]),
                'velocity_ego': [float(velocity[0]), float(velocity[1])],
                'velocity_global': [float(vel_global[0]), float(vel_global[1])],
                'speed': float(speed),
                'motion_state': obj_motion,
                'visible_cameras': visible_cameras,
                'relative_position': f"{longitudinal}-{lateral}",
                'ttc': float(ttc_result['ttc']) if not np.isinf(ttc_result['ttc']) else None,
                'risk_level': ttc_result['risk_level'],
                'closing_speed': float(ttc_result['intermediate'].get('closing_speed', 0))
            })

        return objects

    def _count_objects_by_class(self, objects: List[Dict]) -> Dict[str, int]:
        """Count objects by class/category."""
        counts = defaultdict(int)
        for obj in objects:
            counts[obj['category']] += 1
        return dict(counts)

    def _group_objects_by_camera(self, objects: List[Dict]) -> Dict[str, List[Dict]]:
        """Group objects by camera view."""
        by_camera = {cam: [] for cam in CAMERA_NAMES}
        for obj in objects:
            for cam in obj['visible_cameras']:
                by_camera[cam].append(obj)
        return by_camera


# ============================================================================
# VQA RESULTS LOADER
# ============================================================================

class VQAResultsLoader:
    """Load and parse existing VQA results for risk information."""

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self._cache = {}

    def find_result_file(self, sample_idx: int) -> Optional[str]:
        """Find VQA result file for a sample index."""
        # Pattern: vqa_results_single_frame_*_*_<sample_idx>.json
        pattern = os.path.join(self.results_dir, f"vqa_results_single_frame_*_*_{sample_idx:04d}.json")
        matches = glob.glob(pattern)

        if matches:
            return matches[0]

        # Also try without zero-padding
        pattern = os.path.join(self.results_dir, f"vqa_results_single_frame_*_*_{sample_idx}.json")
        matches = glob.glob(pattern)

        return matches[0] if matches else None

    def load_result(self, sample_idx: int) -> Optional[Dict]:
        """Load VQA result for a sample."""
        if sample_idx in self._cache:
            return self._cache[sample_idx]

        file_path = self.find_result_file(sample_idx)
        if not file_path:
            return None

        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
            self._cache[sample_idx] = result
            return result
        except Exception as e:
            print(f"Error loading VQA result: {e}")
            return None

    def extract_risk_summary(self, sample_idx: int) -> Optional[Dict]:
        """Extract risk information summary from VQA result."""
        result = self.load_result(sample_idx)
        if not result:
            return None

        response = result.get('response', '')

        # Extract key sections from response
        sections = {}

        # Parse "Overall Risk Level" section
        if "Overall Risk Level" in response:
            risk_line = response.split("Overall Risk Level")[-1].split("\n")[0]
            if "Low" in risk_line:
                sections['overall_risk'] = "Low"
            elif "Moderate" in risk_line:
                sections['overall_risk'] = "Moderate"
            elif "High" in risk_line:
                sections['overall_risk'] = "High"

        sections['full_response'] = response
        sections['description'] = result.get('description', '')
        sections['location'] = result.get('location', '')

        return sections


# ============================================================================
# QWEN3-VL INFERENCE MODULE
# ============================================================================

class Qwen3VLInference:
    """Qwen3-VL model for template filtering inference."""

    def __init__(self, model_path: str, loader: 'NuScenesDataLoader',
                 max_new_tokens: int = 1024, resize_factor: int = 2):
        """
        Initialize the Qwen3-VL model for inference.

        Args:
            model_path: Path to the Qwen3-VL model checkpoint
            loader: NuScenesDataLoader instance for image loading
            max_new_tokens: Maximum tokens to generate
            resize_factor: Image resize factor (1/n of original size)
        """
        if not INFERENCE_AVAILABLE:
            raise RuntimeError(
                "Inference dependencies not available. "
                "Please install: pip install torch transformers"
            )

        print("=" * 60)
        print("Initializing Qwen3-VL for Template Filtering")
        print("=" * 60)

        print(f"\nLoading model from: {model_path}")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map='auto',
        )
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        self.loader = loader
        self.max_new_tokens = max_new_tokens
        self.resize_factor = resize_factor
        print("✓ Model loaded successfully\n")

    def prepare_messages(self, sample, prompt: str) -> List[Dict]:
        """
        Prepare messages with 6 camera images and filtering prompt.

        Args:
            sample: NuScenesSample object
            prompt: The filtering prompt text

        Returns:
            Messages formatted for Qwen3-VL
        """
        from PIL import Image

        user_content = []

        # Add 6 camera images in egocentric order (FL, F, FR, RL, R, RR)
        # Rear cameras are horizontally flipped for egocentric consistency
        if self.resize_factor > 1:
            print(f"  Resizing images by factor {self.resize_factor} (1/{self.resize_factor} of original)...")
            for cam_name in CAMERA_NAMES:
                img_path = sample.cameras[cam_name].image_path
                flip = cam_name in REAR_CAMERAS
                img_array = self.loader.load_image(
                    img_path, resize_factor=self.resize_factor, flip_horizontal=flip
                )
                img_pil = Image.fromarray(img_array)
                user_content.append({"type": "image", "image": img_pil})
        else:
            for cam_name in CAMERA_NAMES:
                img_path = sample.cameras[cam_name].image_path
                flip = cam_name in REAR_CAMERAS
                img_array = self.loader.load_image(img_path, flip_horizontal=flip)
                img_pil = Image.fromarray(img_array)
                user_content.append({"type": "image", "image": img_pil})

        # Add the prompt text
        user_content.append({"type": "text", "text": prompt})

        # Create messages with system and user roles
        system_prompt = """You are an expert autonomous driving vision-language model assistant.
You are analyzing 6 camera views from an autonomous vehicle in the following order:
1. Front-left camera (CAM_FRONT_LEFT)
2. Front camera (CAM_FRONT)
3. Front-right camera (CAM_FRONT_RIGHT)
4. Rear-left camera (CAM_BACK_LEFT)
5. Rear camera (CAM_BACK)
6. Rear-right camera (CAM_BACK_RIGHT)

Rear camera images are horizontally flipped for egocentric consistency (left stays left, right stays right).

Your task is to evaluate question templates and select the ones that are applicable to the current scene based on visual analysis and prior knowledge."""

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": user_content
            }
        ]

        return messages

    def run_inference(self, messages: List[Dict]) -> str:
        """
        Run inference on the given messages.

        Args:
            messages: Messages formatted for Qwen3-VL

        Returns:
            Generated text response
        """
        # Prepare inputs
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors='pt'
        )

        inputs = inputs.to(self.model.device)

        # Clear CUDA cache
        torch.cuda.empty_cache()

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0
            )

        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        # Clear cache after generation
        torch.cuda.empty_cache()

        return output_text[0]

    def parse_json_response(self, response: str) -> Optional[Dict]:
        """
        Parse JSON from the model response.

        Args:
            response: Raw text response from the model

        Returns:
            Parsed JSON dictionary or None if parsing fails
        """
        # Try to find JSON in the response
        # First, try to parse the entire response as JSON
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{[^{}]*"category"[^{}]*"selected_templates"[^{}]*\[.*?\][^{}]*\}'
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue

        # Try to find any JSON object in the response
        brace_count = 0
        start_idx = None
        for i, char in enumerate(response):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    try:
                        json_str = response[start_idx:i+1]
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        start_idx = None

        print("Warning: Could not parse JSON from response")
        return None


# ============================================================================
# CATEGORY-SPECIFIC PRIOR KNOWLEDGE BUILDERS
# ============================================================================

def build_observation_prior(scene_data: Dict) -> str:
    """Build prior knowledge for Observation category."""
    lines = ["=== PRIOR KNOWLEDGE: OBSERVATION ==="]
    lines.append("You have access to the following information about objects in the scene:")
    lines.append("")

    # Object counts by class
    lines.append("** Object Counts by Class **")
    class_counts = scene_data['class_counts']
    if class_counts:
        for cls, count in sorted(class_counts.items()):
            lines.append(f"  - {cls}: {count}")
    else:
        lines.append("  - No objects detected")
    lines.append("")

    # Objects list with numbering
    lines.append("** Objects in Scene **")
    if scene_data['objects']:
        for i, obj in enumerate(scene_data['objects'], 1):
            cam_ref = _format_camera_refs(obj['visible_cameras'])
            lines.append(f"  OBJ {i}: {obj['category']} at distance={obj['distance']:.1f}m "
                        f"({obj['relative_position']}) visible in {cam_ref}")
    else:
        lines.append("  - No objects detected")
    lines.append("")

    # Objects by camera view
    lines.append("** Objects by Camera View **")
    for cam_name in CAMERA_NAMES:
        img_num = CAM_TO_IMAGE_NUM[cam_name]
        cam_display = CAMERA_NAME_MAP[cam_name]
        cam_objects = scene_data['objects_by_camera'].get(cam_name, [])
        if cam_objects:
            obj_types = [obj['category'] for obj in cam_objects]
            lines.append(f"  - Image {img_num} ({cam_display}): {len(cam_objects)} objects ({', '.join(set(obj_types))})")
        else:
            lines.append(f"  - Image {img_num} ({cam_display}): No objects")
    lines.append("")

    # Total objects
    lines.append(f"** Total Objects in Scene: {scene_data['total_objects']} **")

    return "\n".join(lines)


def build_identification_prior(scene_data: Dict) -> str:
    """Build prior knowledge for Identification category."""
    lines = ["=== PRIOR KNOWLEDGE: IDENTIFICATION ==="]
    lines.append("You have access to the following object information for identification:")
    lines.append("")

    # List objects with their categories and positions
    lines.append("** Objects and Their Categories **")
    for i, obj in enumerate(scene_data['objects'], 1):
        cam_ref = _format_camera_refs(obj['visible_cameras'])
        lines.append(f"  OBJ {i}: {obj['category']} at distance={obj['distance']:.1f}m "
                    f"({obj['relative_position']}) visible in {cam_ref}")

    if not scene_data['objects']:
        lines.append("  - No objects detected for identification")

    lines.append("")
    lines.append("Note: Visual inspection may reveal additional attributes (color, special markings, etc.)")

    return "\n".join(lines)


def build_attributes_states_prior(scene_data: Dict) -> str:
    """Build prior knowledge for Attributes_and_States category."""
    lines = ["=== PRIOR KNOWLEDGE: ATTRIBUTES AND STATES ==="]
    lines.append("You have access to the following state information:")
    lines.append("")

    # Object motion states
    lines.append("** Object Motion States **")
    for i, obj in enumerate(scene_data['objects'], 1):
        vel_str = f"[vx={obj['velocity_ego'][0]:.2f}, vy={obj['velocity_ego'][1]:.2f}] m/s"
        cam_ref = _format_camera_refs(obj['visible_cameras'])
        lines.append(f"  OBJ {i}: {obj['category']} ({obj['relative_position']}), "
                    f"{obj['motion_state'].upper()}, velocity={vel_str}, speed={obj['speed']:.2f} m/s, "
                    f"visible in {cam_ref}")

    if not scene_data['objects']:
        lines.append("  - No objects detected")

    lines.append("")

    # Ego vehicle state
    ego = scene_data['ego_info']
    lines.append("** Ego Vehicle State **")
    lines.append(f"  - Motion: {ego['motion_state'].upper()}")
    lines.append(f"  - Speed: {ego['speed']:.2f} m/s ({ego['speed_kmh']:.1f} km/h)")
    lines.append(f"  - Command: {ego['driving_command']}")
    lines.append("")

    lines.append("Note: Visual attributes (lights, signals, pedestrian actions) require image inspection.")

    return "\n".join(lines)


def build_spatial_prior(scene_data: Dict) -> str:
    """Build prior knowledge for Spatial_Relationships_and_Occlusion category."""
    lines = ["=== PRIOR KNOWLEDGE: SPATIAL RELATIONSHIPS ==="]
    lines.append("You have access to 3D position data for spatial reasoning:")
    lines.append("")

    # Objects with distances and positions
    lines.append("** Object Positions (Ego-centric) **")
    for i, obj in enumerate(scene_data['objects'], 1):
        pos = obj['position_ego']
        cam_ref = _format_camera_refs(obj['visible_cameras'])
        lines.append(f"  OBJ {i}: {obj['category']}, distance={obj['distance']:.1f}m, "
                    f"position=[x={pos[0]:.1f}m {'ahead' if pos[0] > 0 else 'behind'}, "
                    f"y={pos[1]:.1f}m {'left' if pos[1] > 0 else 'right'}], "
                    f"visible in {cam_ref}")

    if not scene_data['objects']:
        lines.append("  - No objects detected")

    lines.append("")

    # Find closest objects by direction
    lines.append("** Closest Objects by Direction **")
    directions = {'ahead': [], 'behind': [], 'left': [], 'right': []}
    for i, obj in enumerate(scene_data['objects'], 1):
        pos = obj['relative_position']
        for direction in directions:
            if direction in pos:
                directions[direction].append((i, obj))

    for direction, objs in directions.items():
        if objs:
            closest_i, closest = min(objs, key=lambda x: x[1]['distance'])
            lines.append(f"  - {direction.capitalize()}: OBJ {closest_i} ({closest['category']}) at {closest['distance']:.1f}m")
        else:
            lines.append(f"  - {direction.capitalize()}: Clear")

    lines.append("")
    lines.append("Note: Occlusion relationships require visual analysis of camera images.")

    return "\n".join(lines)


def build_traffic_signs_prior(scene_data: Dict) -> str:
    """Build prior knowledge for Traffic_Signs_and_Signals category."""
    lines = ["=== PRIOR KNOWLEDGE: TRAFFIC SIGNS AND SIGNALS ==="]
    lines.append("Limited structured data available for traffic signs/signals.")
    lines.append("")

    # Check for traffic-related objects (keep their global OBJ index)
    traffic_objects = []
    for i, obj in enumerate(scene_data['objects'], 1):
        if 'traffic' in obj['category'].lower() or 'cone' in obj['category'].lower():
            traffic_objects.append((i, obj))

    if traffic_objects:
        lines.append("** Detected Traffic-Related Objects **")
        for i, obj in traffic_objects:
            cam_ref = _format_camera_refs(obj['visible_cameras'])
            lines.append(f"  OBJ {i}: {obj['category']} at distance={obj['distance']:.1f}m, visible in {cam_ref}")
    else:
        lines.append("** No traffic-related objects in 3D annotations **")

    lines.append("")
    lines.append("Note: Traffic signs, signals, and text require visual extraction from camera images.")
    lines.append("The VLM should analyze images to identify traffic lights, signs, and road markings.")

    return "\n".join(lines)


def build_road_markings_prior(scene_data: Dict) -> str:
    """Build prior knowledge for Road_Markings_and_Lane_Configuration category."""
    lines = ["=== PRIOR KNOWLEDGE: ROAD MARKINGS AND LANE CONFIGURATION ==="]
    lines.append("Road marking information requires visual extraction from images.")
    lines.append("")

    # Scene context
    meta = scene_data['scene_meta']
    lines.append("** Scene Context **")
    lines.append(f"  - Location: {meta['location']}")
    lines.append(f"  - Description: {meta['description']}")
    lines.append("")

    lines.append("** Available for Visual Analysis **")
    lines.append("  - Lane line types (solid, dashed, double)")
    lines.append("  - Lane line colors (white, yellow)")
    lines.append("  - Road surface markings (arrows, text, crosswalks)")
    lines.append("  - Lane count and configuration")
    lines.append("")
    lines.append("Note: The VLM should analyze camera images (especially Front view) for lane information.")

    return "\n".join(lines)


def build_risk_assessment_prior(scene_data: Dict, vqa_result: Optional[Dict] = None) -> str:
    """Build prior knowledge for Dynamic_Agents_and_Risk_Assessment category."""
    lines = ["=== PRIOR KNOWLEDGE: DYNAMIC AGENTS AND RISK ASSESSMENT ==="]
    lines.append("Pre-computed TTC and risk metrics are available:")
    lines.append("")

    # Risk summary
    lines.append("** Risk Summary by Object **")
    risk_counts = {'CRITICAL': 0, 'HIGH': 0, 'MODERATE': 0, 'LOW': 0, 'NONE': 0}

    for i, obj in enumerate(scene_data['objects'], 1):
        risk = obj['risk_level']
        risk_counts[risk] += 1
        ttc_str = f"{obj['ttc']:.2f}s" if obj['ttc'] is not None else "∞"
        cam_ref = _format_camera_refs(obj['visible_cameras'])
        vel_str = f"[vx={obj['velocity_ego'][0]:.2f}, vy={obj['velocity_ego'][1]:.2f}] m/s"
        lines.append(f"  OBJ {i}: {obj['category']} [{cam_ref}]")
        lines.append(f"    distance={obj['distance']:.1f}m, velocity={vel_str}, "
                    f"TTC={ttc_str}, Risk={risk}, closing_speed={obj['closing_speed']:.2f}m/s")

    if not scene_data['objects']:
        lines.append("  - No objects detected for risk assessment")

    lines.append("")
    lines.append("** Risk Level Distribution **")
    for level, count in risk_counts.items():
        if count > 0:
            lines.append(f"  - {level}: {count} objects")

    lines.append("")

    # Include VQA result if available
    if vqa_result:
        lines.append("** Previous VQA Risk Analysis Summary **")
        if 'overall_risk' in vqa_result:
            lines.append(f"  - Overall Risk Level: {vqa_result['overall_risk']}")
        lines.append(f"  - Scene: {vqa_result.get('description', 'N/A')}")

    return "\n".join(lines)


def build_planning_prior(scene_data: Dict) -> str:
    """Build prior knowledge for Right_of_Way_and_Planning category."""
    lines = ["=== PRIOR KNOWLEDGE: RIGHT OF WAY AND PLANNING ==="]
    lines.append("Ego vehicle intent and scene context for planning:")
    lines.append("")

    # Ego intent
    ego = scene_data['ego_info']
    lines.append("** Ego Vehicle Status **")
    lines.append(f"  - Driving Command: {ego['driving_command']}")
    lines.append(f"  - Current Speed: {ego['speed']:.2f} m/s ({ego['speed_kmh']:.1f} km/h)")
    lines.append(f"  - Motion State: {ego['motion_state']}")
    lines.append("")

    # Nearby objects affecting planning
    lines.append("** Nearby Objects (within 15m) **")
    nearby = [(i, obj) for i, obj in enumerate(scene_data['objects'], 1) if obj['distance'] < 15]
    for i, obj in nearby:
        ttc_str = f"{obj['ttc']:.2f}s" if obj['ttc'] is not None else "∞"
        cam_ref = _format_camera_refs(obj['visible_cameras'])
        lines.append(f"  OBJ {i}: {obj['category']} at distance={obj['distance']:.1f}m "
                    f"({obj['relative_position']}), TTC={ttc_str}, visible in {cam_ref}")

    if not nearby:
        lines.append("  - No objects within 15m")

    lines.append("")
    lines.append("Note: Right-of-way rules and lane restrictions require visual analysis of signs/signals.")

    return "\n".join(lines)


def build_environmental_prior(scene_data: Dict) -> str:
    """Build prior knowledge for Environmental_and_Sensor_Conditions category."""
    lines = ["=== PRIOR KNOWLEDGE: ENVIRONMENTAL CONDITIONS ==="]
    lines.append("Environmental information from scene metadata and visual analysis:")
    lines.append("")

    # Scene context
    meta = scene_data['scene_meta']
    lines.append("** Scene Metadata **")
    lines.append(f"  - Location: {meta['location']}")
    lines.append(f"  - Description: {meta['description']}")
    lines.append("")

    lines.append("** Requires Visual Analysis **")
    lines.append("  - Weather conditions (sunny, cloudy, rainy, foggy)")
    lines.append("  - Time of day (day, night, dusk, dawn)")
    lines.append("  - Road surface condition (dry, wet, icy)")
    lines.append("  - Visibility quality (clear, moderate, poor)")
    lines.append("  - Sensor/camera quality (clean, obscured)")
    lines.append("")
    lines.append("Note: The VLM should analyze all 6 camera views to determine environmental conditions.")

    return "\n".join(lines)


def build_causal_prior(scene_data: Dict, vqa_result: Optional[Dict] = None) -> str:
    """Build prior knowledge for Causal_and_Hypothetical_Reasoning category."""
    lines = ["=== PRIOR KNOWLEDGE: CAUSAL AND HYPOTHETICAL REASONING ==="]
    lines.append("Aggregated scene information for causal/hypothetical analysis:")
    lines.append("")

    # Scene summary
    meta = scene_data['scene_meta']
    ego = scene_data['ego_info']

    lines.append("** Situation Summary **")
    lines.append(f"  - Location: {meta['location']}")
    lines.append(f"  - Scene: {meta['description']}")
    lines.append(f"  - Ego State: {ego['motion_state']} ({ego['speed_kmh']:.1f} km/h), command={ego['driving_command']}")
    lines.append(f"  - Total Objects: {scene_data['total_objects']}")
    lines.append("")

    # Dynamic objects for causal reasoning
    lines.append("** Dynamic Objects (Moving) **")
    moving = [(i, obj) for i, obj in enumerate(scene_data['objects'], 1) if obj['motion_state'] == 'moving']
    for i, obj in moving:
        cam_ref = _format_camera_refs(obj['visible_cameras'])
        lines.append(f"  OBJ {i}: {obj['category']} at distance={obj['distance']:.1f}m, "
                    f"speed={obj['speed']:.2f}m/s, "
                    f"heading {'toward' if obj['closing_speed'] > 0.5 else 'away/parallel'}, "
                    f"visible in {cam_ref}")

    if not moving:
        lines.append("  - No moving objects detected")

    lines.append("")

    # Physics-based info
    lines.append("** Physics-Based Information **")
    if ego['speed'] > 0.5:
        stopping_dist = (ego['speed'] ** 2) / (2 * 3.5)  # Assuming 3.5 m/s² deceleration
        lines.append(f"  - Estimated stopping distance (comfortable): {stopping_dist:.1f}m")
    else:
        lines.append("  - Ego is nearly stationary, minimal stopping distance needed")

    lines.append("")
    lines.append("Note: Causal and hypothetical reasoning requires understanding scene dynamics from images.")

    return "\n".join(lines)


# Map categories to prior knowledge builders
PRIOR_BUILDERS = {
    "Observation": build_observation_prior,
    "Identification": build_identification_prior,
    "Attributes_and_States": build_attributes_states_prior,
    "Spatial_Relationships_and_Occlusion": build_spatial_prior,
    "Traffic_Signs_and_Signals": build_traffic_signs_prior,
    "Road_Markings_and_Lane_Configuration": build_road_markings_prior,
    "Dynamic_Agents_and_Risk_Assessment": build_risk_assessment_prior,
    "Right_of_Way_and_Planning": build_planning_prior,
    "Environmental_and_Sensor_Conditions": build_environmental_prior,
    "Causal_and_Hypothetical_Reasoning": build_causal_prior
}


# ============================================================================
# QUESTION BANK AND TEMPLATE FILTERING
# ============================================================================

def load_question_bank(path: str) -> Dict:
    """Load the question bank JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def get_category_templates(question_bank: Dict, category: str) -> List[Dict]:
    """Get templates for a specific category.

    Supports both legacy format (templates directly on category) and
    v3 format (templates nested inside sub_categories).
    """
    for cat in question_bank['categories']:
        if cat['category'] == category:
            # v3 format: templates are nested inside sub_categories
            if 'sub_categories' in cat:
                templates = []
                for sub_cat in cat['sub_categories']:
                    templates.extend(sub_cat.get('templates', []))
                return templates
            # Legacy format: templates directly on category
            return cat.get('templates', [])
    return []


def format_templates_for_batch_validation(templates: List[Dict], start_idx: int) -> str:
    """
    Format a batch of templates for placeholder validation.

    For each template, list:
    - The template string with placeholders highlighted
    - All possible values for each placeholder
    - What needs to be verified in the scene
    - Ego-centric note (if present) to guide ego-perspective evaluation
    """
    lines = []
    for i, t in enumerate(templates):
        global_idx = start_idx + i + 1  # 1-indexed global template number
        template_str = t['template']
        placeholders = t.get('placeholders', {})
        answer_type = t.get('answer_type', 'open_ended')
        template_id = t.get('template_id', '')
        ego_note = t.get('ego_centric_note', '')

        header = f"--- Template #{global_idx}"
        if template_id:
            header += f" [{template_id}]"
        header += " ---"
        lines.append(header)
        lines.append(f"Question: {template_str}")
        lines.append(f"Answer Type: {answer_type}")

        if ego_note:
            lines.append(f"Ego-Centric Guidance: {ego_note}")

        if placeholders:
            lines.append("Placeholders to verify:")
            for tag, values in placeholders.items():
                if isinstance(values, list):
                    lines.append(f"  <{tag}>: {values}")
                else:
                    lines.append(f"  <{tag}>: [{values}]")
            lines.append("Verification task: Check which placeholder values can be grounded in the scene from the ego-vehicle's perspective.")
        else:
            lines.append("No placeholders - verify if the question is answerable for this scene from the ego-vehicle's perspective.")

        lines.append("")

    return "\n".join(lines)


def build_batch_validation_prompt(category: str, prior_knowledge: str,
                                   templates: List[Dict], start_idx: int,
                                   batch_num: int, total_batches: int) -> str:
    """
    Build prompt for VLM to validate a batch of templates.

    The VLM should:
    1. Inspect each template's placeholder tags
    2. Check if the placeholder values can be grounded in the scene (images + prior)
    3. Report which templates are applicable and with which placeholder values
    """

    prompt = f"""You are validating ego-centric question templates for a Visual Question Answering (VQA) dataset.

TASK: For each template below, determine if it is APPLICABLE to the current scene by verifying whether the placeholder values can be grounded in the provided images and prior information, all from the ego-vehicle's perspective.

You are viewing 6 camera images (Front-left, Front, Front-right, Rear-left, Rear, Rear-right) from the ego-vehicle. Rear camera images are horizontally flipped for egocentric consistency (left stays left, right stays right from the driver's viewpoint).

{prior_knowledge}

=== TEMPLATES TO VALIDATE (Batch {batch_num}/{total_batches}, Category: {category}) ===

{format_templates_for_batch_validation(templates, start_idx)}

=== VALIDATION CRITERIA (EGO-CENTRIC) ===

**APPLICABLE**: A template is APPLICABLE if AT LEAST ONE semantically consistent combination of placeholder values can be grounded in the provided images and prior information from the ego-vehicle's perspective.

**EGO-CENTRIC GROUNDING**: All spatial terms (e.g., "ahead", "left", "adjacent lane", "behind") must be interpreted relative to the ego-vehicle's position and heading. An object or feature is relevant only if it pertains to the ego-vehicle's driving context — its current lane, intended path, surrounding traffic, or applicable signals/signs.

**EXISTENCE**: A placeholder value is considered to EXIST only if it can be verified from the 6-view images AND/OR is explicitly supported by the prior information. When "Ego-Centric Guidance" is provided for a template, use it to determine which camera views and spatial zones are relevant for verification.

**NOT_APPLICABLE**: Mark as NOT_APPLICABLE if:
- None of the placeholder values exist in the scene from the ego-vehicle's perspective
- The question cannot be meaningfully answered given the ego-vehicle's current situation
- The scene context is irrelevant to the ego-vehicle's driving context
- The grounding between placeholder values and the ego-vehicle's perspective is ambiguous or unsupported

=== VALIDATION PROCESS ===

For EACH template:

1. **If the template HAS placeholders:**
   - Inspect each placeholder tag (e.g., <object>, <direction>, <spatial_relation>)
   - For each possible value, verify if it can be grounded in the images or prior from the ego-vehicle's viewpoint
   - Use the "Ego-Centric Guidance" note (if provided) to determine which camera views and spatial references are relevant
   - List ONLY the values that are verifiably present relative to the ego-vehicle
   - Template is applicable if at least one valid combination exists

2. **If the template has NO placeholders:**
   - Verify if the question can be meaningfully answered for this scene from the ego-vehicle's perspective
   - Check if required objects/conditions are verifiably present in the ego-vehicle's driving context

=== OUTPUT FORMAT ===

Respond with a JSON object:
{{
    "batch_results": [
        {{
            "template_idx": <global_template_number>,
            "applicable": true/false,
            "valid_placeholders": {{
                "<tag1>": ["verified_value1", "verified_value2"],
                "<tag2>": ["verified_value"]
            }},
            "reason": "<brief explanation of what was verified from the ego-vehicle's perspective>"
        }},
        ...
    ]
}}

- Include ALL templates from this batch in your response
- For templates without placeholders, set "valid_placeholders" to {{}}
- Only include placeholder values that are VERIFIABLY grounded in the scene from the ego-vehicle's perspective

Output ONLY the JSON object, no additional text."""

    return prompt


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate category-specific question templates for VQA dataset"
    )
    parser.add_argument(
        "--sample_idx", type=int, required=True,
        help="Sample index from nuScenes dataset"
    )
    parser.add_argument(
        "--category", type=str, required=True,
        choices=VALID_CATEGORIES + ["all"],
        help="Question category to process, or 'all' for all categories"
    )
    parser.add_argument(
        "--pkl_path", type=str,
        default=os.environ.get("NUSCENES_PKL_PATH", "nuscenes2d_ego_temporal_infos_val.pkl"),
        help="Path to nuScenes pkl file"
    )
    parser.add_argument(
        "--question_bank", type=str,
        default=os.environ.get("QUESTION_BANK_PATH", "question_bank.json"),
        help="Path to question bank JSON"
    )
    parser.add_argument(
        "--vqa_results_dir", type=str,
        default="vqa_results",
        help="Directory containing VQA results files"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="qa_outputs",
        help="Output directory for generated prompts"
    )
    parser.add_argument(
        "--filter_distance", type=float, default=20.0,
        help="Maximum distance (in meters) from ego vehicle to include objects. Default 20.0m"
    )
    parser.add_argument(
        "--rear_filter", type=float, default=None,
        help="Maximum distance (in meters) for non-vehicle objects behind the ego vehicle. "
             "If not set, uses --filter_distance for all directions."
    )
    parser.add_argument(
        "--show_prompt", action="store_true",
        help="Print the generated prompt instead of saving"
    )
    parser.add_argument(
        "--run_inference", action="store_true",
        help="Run Qwen3-VL inference to filter templates"
    )
    parser.add_argument(
        "--model_path", type=str,
        default="./ckpts/qwen3_vl_235b_a22b_instruct",
        help="Path to Qwen3-VL model checkpoint"
    )
    parser.add_argument(
        "--resize_factor", type=int, default=2,
        help="Image resize factor (1/n of original size)"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=1024,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=5,
        help="Number of templates to process per batch (to avoid GPU OOM)"
    )

    args = parser.parse_args()

    # Initialize components
    print(f"Loading nuScenes data from {args.pkl_path}...")
    loader = NuScenesDataLoader(args.pkl_path)
    analyzer = SceneAnalyzer(loader)
    vqa_loader = VQAResultsLoader(args.vqa_results_dir)

    # Initialize inference module if needed
    vlm_inference = None
    if args.run_inference:
        if not INFERENCE_AVAILABLE:
            print("Error: Inference dependencies not available.")
            print("Please install: pip install torch transformers")
            sys.exit(1)
        vlm_inference = Qwen3VLInference(
            model_path=args.model_path,
            loader=loader,
            max_new_tokens=args.max_new_tokens,
            resize_factor=args.resize_factor
        )

    # Load question bank
    print(f"Loading question bank from {args.question_bank}...")
    question_bank = load_question_bank(args.question_bank)

    # Analyze sample with distance filtering
    print(f"\nAnalyzing sample {args.sample_idx}...")
    print(f"  Filter distance: {args.filter_distance}m")
    if args.rear_filter is not None:
        print(f"  Rear filter distance (non-vehicles): {args.rear_filter}m")
    scene_data = analyzer.analyze_sample(
        args.sample_idx,
        max_distance=args.filter_distance,
        rear_filter_distance=args.rear_filter
    )

    # Get sample object for inference
    sample = loader.get_sample(args.sample_idx)

    # Load VQA result if available
    vqa_result = vqa_loader.extract_risk_summary(args.sample_idx)
    if vqa_result:
        print(f"  Loaded VQA result (Overall Risk: {vqa_result.get('overall_risk', 'N/A')})")

    # Print scene summary
    print(f"\n=== Scene Summary ===")
    print(f"  Location: {scene_data['scene_meta']['location']}")
    print(f"  Description: {scene_data['scene_meta']['description']}")
    print(f"  Total Objects: {scene_data['total_objects']}")
    print(f"  Class Counts: {scene_data['class_counts']}")
    print(f"  Ego Speed: {scene_data['ego_info']['speed_kmh']:.1f} km/h ({scene_data['ego_info']['motion_state']})")

    # Process categories
    categories = VALID_CATEGORIES if args.category == "all" else [args.category]

    os.makedirs(args.output_dir, exist_ok=True)

    BATCH_SIZE = args.batch_size  # Process templates in batches to avoid GPU OOM

    results = {}
    inference_results = {}  # Store ALL applicable templates from inference

    for category in categories:
        print(f"\n{'='*60}")
        print(f"Processing: {category}")
        print(f"{'='*60}")

        # Build prior knowledge
        builder = PRIOR_BUILDERS[category]
        if category in ["Dynamic_Agents_and_Risk_Assessment", "Causal_and_Hypothetical_Reasoning"]:
            prior_knowledge = builder(scene_data, vqa_result)
        else:
            prior_knowledge = builder(scene_data)

        # Get templates
        templates = get_category_templates(question_bank, category)
        num_templates = len(templates)
        num_batches = (num_templates + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Templates available: {num_templates}")
        print(f"  Processing in {num_batches} batches of {BATCH_SIZE}")

        results[category] = {
            'category': category,
            'num_templates': num_templates,
            'prior_knowledge': prior_knowledge,
            'num_batches': num_batches
        }

        # Run inference if requested - process in batches
        if vlm_inference is not None:
            print(f"\n  Running Qwen3-VL inference for {category}...")

            all_applicable = []  # Collect ALL applicable templates
            all_batch_responses = []  # Store raw responses for debugging

            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, num_templates)
                batch_templates = templates[start_idx:end_idx]

                print(f"\n  --- Batch {batch_idx + 1}/{num_batches} (templates {start_idx + 1}-{end_idx}) ---")

                # Build batch validation prompt
                prompt = build_batch_validation_prompt(
                    category=category,
                    prior_knowledge=prior_knowledge,
                    templates=batch_templates,
                    start_idx=start_idx,
                    batch_num=batch_idx + 1,
                    total_batches=num_batches
                )

                if args.show_prompt and batch_idx == 0:
                    print(f"\n--- Sample Prompt (Batch 1) ---")
                    print(prompt[:2000] + "..." if len(prompt) > 2000 else prompt)
                    print(f"\n--- End of sample prompt ---")

                # Prepare messages with images and prompt
                messages = vlm_inference.prepare_messages(sample, prompt)

                # Run inference
                response = vlm_inference.run_inference(messages)
                all_batch_responses.append({
                    'batch_idx': batch_idx + 1,
                    'template_range': f"{start_idx + 1}-{end_idx}",
                    'response': response
                })

                # Parse the JSON response
                parsed = vlm_inference.parse_json_response(response)

                if parsed and 'batch_results' in parsed:
                    batch_results = parsed['batch_results']

                    # Count applicable in this batch
                    applicable_count = sum(1 for r in batch_results if r.get('applicable', False))
                    print(f"      ✓ Parsed: {applicable_count}/{len(batch_results)} applicable")

                    # Collect applicable templates with their verified placeholders
                    for result in batch_results:
                        if result.get('applicable', False):
                            template_idx = int(result.get('template_idx', 0)) - 1  # Convert to 0-indexed
                            if 0 <= template_idx < num_templates:
                                original_template = templates[template_idx]
                                all_applicable.append({
                                    'template_idx': template_idx + 1,  # 1-indexed for output
                                    'template': original_template['template'],
                                    'answer_type': original_template.get('answer_type', 'open_ended'),
                                    'original_placeholders': original_template.get('placeholders', {}),
                                    'valid_placeholders': result.get('valid_placeholders', {}),
                                    'reason': result.get('reason', '')
                                })
                else:
                    print(f"      ✗ Failed to parse batch {batch_idx + 1}")

            # Store results for this category
            print(f"\n  === Category Summary: {len(all_applicable)}/{num_templates} templates applicable ===")

            # Format applicable templates for output
            category_applicable = {}
            for i, item in enumerate(all_applicable, 1):
                q_key = f"q_{i:02d}"
                category_applicable[q_key] = {
                    'template_idx': item['template_idx'],
                    'template': item['template'],
                    'answer_type': item['answer_type'],
                    'valid_placeholders': item['valid_placeholders'],
                    'reason': item['reason']
                }

            inference_results[category] = category_applicable
            results[category]['applicable_templates'] = all_applicable
            results[category]['num_applicable'] = len(all_applicable)
            results[category]['batch_responses'] = all_batch_responses
        else:
            # No inference - just show prompt structure
            if args.show_prompt:
                sample_prompt = build_batch_validation_prompt(
                    category=category,
                    prior_knowledge=prior_knowledge,
                    templates=templates[:BATCH_SIZE],
                    start_idx=0,
                    batch_num=1,
                    total_batches=num_batches
                )
                print(f"\n--- Sample Prompt for {category} (Batch 1) ---")
                print(sample_prompt)
                print(f"\n--- End of sample prompt ---")

    # Save results
    output_file = os.path.join(args.output_dir, f"sample_{args.sample_idx}_qa_summary.json")

    # Save summary
    summary = {
        'sample_idx': args.sample_idx,
        'scene_meta': scene_data['scene_meta'],
        'ego_info': scene_data['ego_info'],
        'class_counts': scene_data['class_counts'],
        'total_objects': scene_data['total_objects'],
        'categories': {
            cat: {
                'num_templates': data['num_templates'],
                'num_batches': data.get('num_batches', 0),
                'num_applicable': data.get('num_applicable', 0)
            }
            for cat, data in results.items()
        }
    }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Summary saved to: {output_file}")
    print(f"{'='*60}")

    # Save inference results (applicable templates) in the requested format
    if vlm_inference is not None and inference_results:
        # Format: {"sample_idx": {"<category>": {"q_01": {...}, ...}}}
        final_output = {
            args.sample_idx: inference_results
        }

        applicable_file = os.path.join(args.output_dir, f"sample_{args.sample_idx}_applicable_questions.json")
        with open(applicable_file, 'w') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
        print(f"Applicable questions saved to: {applicable_file}")

        # Also save detailed inference results with raw responses for debugging
        detailed_file = os.path.join(args.output_dir, f"sample_{args.sample_idx}_inference_detailed.json")
        detailed_output = {
            'sample_idx': args.sample_idx,
            'scene_meta': scene_data['scene_meta'],
            'timestamp': datetime.now().isoformat(),
            'model_path': args.model_path,
            'resize_factor': args.resize_factor,
            'batch_size': BATCH_SIZE,
            'categories': {}
        }

        for cat, data in results.items():
            detailed_output['categories'][cat] = {
                'num_templates': data['num_templates'],
                'num_batches': data.get('num_batches', 0),
                'num_applicable': data.get('num_applicable', 0),
                'applicable_templates': data.get('applicable_templates', []),
                'batch_responses': data.get('batch_responses', [])
            }

        with open(detailed_file, 'w') as f:
            json.dump(detailed_output, f, indent=4, ensure_ascii=False)
        print(f"Detailed inference results saved to: {detailed_file}")

        # Print summary of applicable templates
        print(f"\n{'='*60}")
        print("INFERENCE SUMMARY - APPLICABLE TEMPLATES")
        print(f"{'='*60}")
        total_applicable = 0
        total_templates = 0
        for cat, questions in inference_results.items():
            cat_data = results[cat]
            num_applicable = len(questions)
            num_total = cat_data['num_templates']
            total_applicable += num_applicable
            total_templates += num_total
            print(f"\n{cat}: {num_applicable}/{num_total} applicable")
            for q_key, q_data in list(questions.items())[:3]:  # Show first 3
                template_preview = q_data['template'][:50] + "..." if len(q_data['template']) > 50 else q_data['template']
                print(f"  {q_key}: {template_preview}")
            if len(questions) > 3:
                print(f"  ... and {len(questions) - 3} more")

        print(f"\n{'='*60}")
        print(f"TOTAL: {total_applicable}/{total_templates} templates applicable across all categories")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
