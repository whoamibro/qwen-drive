"""
Virtual Camera Unification — custom demo dataset → nuScenes-format conversion.

Single source of truth for the crop/resize math that maps each of the six
custom cameras into a COMMON virtual pinhole camera. Both the crop
verification visualizer (`custom_dataset.crop_verifier`) and the batch image
converter import from here — the formulas must not be duplicated elsewhere.

Methodology (see manifest/2026-08-13_custom_dataset_demo_adaptation_plan.md):
    All source images are pinhole-rectified (kbm model with k1..k4 == 0), so
    "equal object scale" across views means equal effective focal length per
    axis. We define one virtual target camera (W_t x H_t, isotropic focal
    f_t, principal point at the output center). Because source and target
    share the optical axis, the warp degenerates to an axis-aligned subpixel
    crop centered on the source principal point, followed by a per-axis
    resize:

        crop_w = W_t * fx / f_t        crop_h = H_t * fy / f_t
        x1 = cx - crop_w / 2           y1 = cy - crop_h / 2

    Feasibility: the crop must lie inside the source image, i.e.
    f_t >= max(W_t * fx / W_src, H_t * fy / H_src) for every camera.

Pitch leveling: the physical cameras are NOT level — measured against the
road plane (see `measure_camera_orientation.py`), the front-side cams point
~9.5 deg UP, the rear-side cams ~10 deg DOWN, and the rear-center a strong
~27 deg DOWN (the front is level). Each virtual camera is a TRUE ROTATION
of its source camera about the optical center: target pixels map to source
pixels through the rectification homography H = K_src·R_x(-e)·K_t^-1
(NOT a shifted crop — an off-axis crop leaves the optical axis tilted and
stretches rows asymmetrically, badly at the rear cam's 27 deg). The
correction clamps to the largest angle whose warped footprint stays inside
the source image; the residual tilt is reported in `crop_metadata`.
Measured elevations are persisted in camera_orientation.json and loaded via
`load_camera_elevations()`.

Intrinsics are read STRICTLY from maps/calib_tuned/<channel>.yaml (fx, fy,
cx, cy). The yaml's `hfov` field is known to be wrong for the side cameras
and is never used — all angles are derived from the intrinsics. The
`vcs_extrinsic` blocks are nominal design values (CAM_BACK's is wrong);
true orientations come from the tuned `lcs_extrinsic` + trajectory.

Usage:
    from custom_dataset.virtual_camera import (
        DEFAULT_TARGET, load_clip_calibs, compute_crop_box, convert_image,
    )
    calibs = load_clip_calibs("/path/to/clip")          # {nusc_cam: CameraCalib}
    box = compute_crop_box(calibs["CAM_FRONT"])          # CropBox at DEFAULT_TARGET
    out = convert_image(img, calibs["CAM_FRONT"])        # 1600x900 PIL image
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Custom channel -> nuScenes camera name. The 7th channel (h30 front zoom)
# is intentionally excluded — the trained model uses 6 views.
CHANNEL_TO_NUSCENES = {
    '2160p_h120_front':           'CAM_FRONT',
    '1536p_h100_side_frontleft':  'CAM_FRONT_LEFT',
    '1536p_h100_side_frontright': 'CAM_FRONT_RIGHT',
    '1536p_h100_side_rearleft':   'CAM_BACK_LEFT',
    '1536p_h100_side_rearright':  'CAM_BACK_RIGHT',
    '1536p_h100_side_rearcenter': 'CAM_BACK',
}
NUSCENES_TO_CHANNEL = {v: k for k, v in CHANNEL_TO_NUSCENES.items()}

# Training-pipeline egocentric view order (matches sft_prompt_builder.CAMERA_ORDER).
CAMERA_ORDER = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',
    'CAM_BACK',
    'CAM_BACK_RIGHT',
]

# Rear-facing views are horizontally flipped for display/inference, matching
# the training pipeline's egocentric convention (sft_model_tester.REAR_CAMERAS).
REAR_CAMERAS = {'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'}

CALIB_SUBDIR = os.path.join('maps', 'calib_tuned')


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetCamera:
    """The common virtual pinhole camera all six views are warped into.

    f_t = 1100 gives 72.1 x 44.5 deg per view — close to the nuScenes 70 deg
    cameras the model was trained on, and small enough that the 55-deg-apart
    side cams overlap the front by ~17 deg instead of ~44 deg (f_t=684), while
    leaving vertical room for the pitch-leveling shifts (rear-center's 27.4
    deg down-tilt levels to a ~3.4 deg residual).
    """
    width: int = 1600
    height: int = 900
    focal: float = 1100.0  # isotropic; principal point at (width/2, height/2)

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.width / (2.0 * self.focal)))

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.height / (2.0 * self.focal)))

    def intrinsic_matrix(self) -> List[List[float]]:
        """3x3 K of the virtual camera (for the nuScenes pkl / projections)."""
        return [
            [self.focal, 0.0, self.width / 2.0],
            [0.0, self.focal, self.height / 2.0],
            [0.0, 0.0, 1.0],
        ]


DEFAULT_TARGET = TargetCamera()

# nuScenes is deliberately asymmetric: CAM_BACK is the one wide camera
# (fx=809.22 at 1600px -> 89.3 deg) while the other five are ~70 deg lenses
# (fx=1266.42 -> 64.6 deg). The SFT model was trained with that per-role
# scale ratio, so the virtual CAM_BACK uses a proportionally shorter focal:
#   f_back = f_base * 809.22 / 1266.42  (= 703 at the default f_base=1100,
#   -> 97.4 deg HFOV, within the rear source's ~101.5 deg).
NUSCENES_BACK_FOCAL_RATIO = 809.22 / 1266.42


def target_for_camera(cam: str,
                      base: TargetCamera = DEFAULT_TARGET,
                      back_focal: Optional[float] = None) -> TargetCamera:
    """Per-camera virtual target: CAM_BACK gets the nuScenes-ratio wide
    focal (or an explicit `back_focal`); all other cameras use `base`.
    Pass back_focal=base.focal to force a uniform-FOV rig."""
    if cam == 'CAM_BACK':
        f = back_focal if back_focal else base.focal * NUSCENES_BACK_FOCAL_RATIO
        return TargetCamera(width=base.width, height=base.height, focal=f)
    return base


@dataclass(frozen=True)
class CameraCalib:
    """Intrinsics (+ raw extrinsics blocks) of one source camera."""
    channel: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    # Raw extrinsic blocks kept verbatim for the later pkl-building phase
    # (quaternion qw..qz + translation tx..tz dicts); None if absent.
    vcs_extrinsic: Optional[dict] = None
    lcs_extrinsic: Optional[dict] = None
    yaml_path: str = ''

    @property
    def hfov_deg(self) -> float:
        """Full horizontal FOV derived from intrinsics (NOT the yaml field)."""
        return math.degrees(math.atan(self.cx / self.fx)
                            + math.atan((self.width - self.cx) / self.fx))

    @property
    def vfov_deg(self) -> float:
        return math.degrees(math.atan(self.cy / self.fy)
                            + math.atan((self.height - self.cy) / self.fy))


@dataclass(frozen=True)
class CropBox:
    """Subpixel crop rectangle in source-image coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def in_bounds(self, img_width: int, img_height: int) -> bool:
        return (self.x1 >= 0.0 and self.y1 >= 0.0
                and self.x2 <= img_width and self.y2 <= img_height)

    def as_tuple(self):
        return (self.x1, self.y1, self.x2, self.y2)


# ---------------------------------------------------------------------------
# Calibration loading
# ---------------------------------------------------------------------------

def load_calib(yaml_path: str) -> CameraCalib:
    """Parse one calib_tuned yaml into a CameraCalib (intrinsics from fx/fy/
    cx/cy only — the yaml's `hfov` field is ignored by design)."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    intr = data['intrinsic']
    return CameraCalib(
        channel=data.get('channel', os.path.splitext(os.path.basename(yaml_path))[0]),
        width=int(data['width']),
        height=int(data['height']),
        fx=float(intr['fx']),
        fy=float(intr['fy']),
        cx=float(intr['cx']),
        cy=float(intr['cy']),
        vcs_extrinsic=data.get('vcs_extrinsic'),
        lcs_extrinsic=data.get('lcs_extrinsic'),
        yaml_path=yaml_path,
    )


def load_clip_calibs(clip_dir: str) -> Dict[str, CameraCalib]:
    """Load the 6 used cameras' calibs for one clip.

    Returns {nuscenes_cam_name: CameraCalib}, reading
    <clip_dir>/maps/calib_tuned/<channel>.yaml for each mapped channel.
    Raises FileNotFoundError with the missing path if a yaml is absent.
    """
    calib_dir = os.path.join(clip_dir, CALIB_SUBDIR)
    out = {}
    for channel, nusc_name in CHANNEL_TO_NUSCENES.items():
        path = os.path.join(calib_dir, f'{channel}.yaml')
        if not os.path.isfile(path):
            raise FileNotFoundError(f'calib yaml not found: {path}')
        out[nusc_name] = load_calib(path)
    return out


# ---------------------------------------------------------------------------
# Measured camera orientation (written by measure_camera_orientation.py)
# ---------------------------------------------------------------------------

ORIENTATION_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'camera_orientation.json')


def load_camera_elevations(path: str = ORIENTATION_JSON) -> Dict[str, float]:
    """Measured optical-axis elevation per camera (deg, + = pointing up),
    road-relative. Returns {} when the file is absent — callers then apply
    no pitch correction. Regenerate with:
        python -m custom_dataset.measure_camera_orientation
    """
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return dict(json.load(f).get('elevation_deg', {}))


# ---------------------------------------------------------------------------
# Warp math (the formulas — defined here and nowhere else)
#
# The virtual camera is a TRUE ROTATION of the source camera about its
# optical center (pitch, about the camera's horizontal axis): the mapping
# from target pixels to source pixels is the rectification homography
#     H = K_src · R_x(alpha) · K_t^-1,      alpha = -elevation
# (camera coords are x-right / y-down / z-forward, so leveling a camera
# whose axis points DOWN by |e| means rotating by alpha = +|e| about x).
# A plain principal-point-shifted crop is NOT equivalent for large angles —
# it produces an off-axis view whose rows are asymmetrically stretched
# (~45% top-vs-bottom error at the rear cam's 27 deg) — so the homography
# is applied whenever the elevation is non-negligible; at elevation 0 it
# degenerates to the axis-aligned crop+resize, which is kept as a fast
# exact path.
# ---------------------------------------------------------------------------

_ZERO_ELEVATION_EPS = 1e-3   # deg; below this the plain crop path is exact


def _homography_out_to_src(calib: CameraCalib, target: TargetCamera,
                           alpha_deg: float):
    """3x3 H mapping target pixel (u', v') -> source pixel (u, v):
    H = K_src · R_x(alpha) · K_t^-1 (row-major nested lists)."""
    a = math.radians(alpha_deg)
    c, s = math.cos(a), math.sin(a)
    ft = target.focal
    tx, ty = target.width / 2.0, target.height / 2.0
    # M = R_x(alpha) · K_t^-1  (K_t^-1 = [[1/ft,0,-tx/ft],[0,1/ft,-ty/ft],[0,0,1]])
    m = [
        [1.0 / ft, 0.0, -tx / ft],
        [0.0, c / ft, -c * ty / ft - s],
        [0.0, s / ft, -s * ty / ft + c],
    ]
    # H = K_src · M
    fx, fy, cx, cy = calib.fx, calib.fy, calib.cx, calib.cy
    return [
        [fx * m[0][0] + cx * m[2][0], fx * m[0][1] + cx * m[2][1], fx * m[0][2] + cx * m[2][2]],
        [fy * m[1][0] + cy * m[2][0], fy * m[1][1] + cy * m[2][1], fy * m[1][2] + cy * m[2][2]],
        [m[2][0], m[2][1], m[2][2]],
    ]


def _map_point(H, u: float, v: float):
    w = H[2][0] * u + H[2][1] * v + H[2][2]
    return ((H[0][0] * u + H[0][1] * v + H[0][2]) / w,
            (H[1][0] * u + H[1][1] * v + H[1][2]) / w)


def _quad_for_alpha(calib: CameraCalib, target: TargetCamera,
                    alpha_deg: float):
    """Target image corners (TL, TR, BR, BL) mapped into source coords."""
    H = _homography_out_to_src(calib, target, alpha_deg)
    W, Ht = target.width, target.height
    return [_map_point(H, u, v) for u, v in ((0, 0), (W, 0), (W, Ht), (0, Ht))]


def _quad_in_bounds(quad, calib: CameraCalib) -> bool:
    return all(0.0 <= x <= calib.width and 0.0 <= y <= calib.height
               for x, y in quad)


def applied_elevation(calib: CameraCalib,
                      target: TargetCamera = DEFAULT_TARGET,
                      elevation_deg: float = 0.0) -> float:
    """The pitch correction actually achievable (deg). Equals elevation_deg
    when the fully-leveled warp quad fits inside the source image; otherwise
    the largest fraction of it that fits (found by bisection). The residual
    is elevation_deg - applied."""
    if abs(elevation_deg) < _ZERO_ELEVATION_EPS:
        return 0.0
    if _quad_in_bounds(_quad_for_alpha(calib, target, -elevation_deg), calib):
        return elevation_deg
    lo, hi = 0.0, 1.0     # fraction of the requested correction
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if _quad_in_bounds(
                _quad_for_alpha(calib, target, -elevation_deg * mid), calib):
            lo = mid
        else:
            hi = mid
    return elevation_deg * lo


def applied_alpha(calib: CameraCalib,
                  target: TargetCamera = DEFAULT_TARGET,
                  elevation_deg: float = 0.0,
                  extra_pitch_deg: float = 0.0) -> float:
    """Total up-rotation actually applied (deg, + = up): the auto-leveling
    correction (clamped to the source-data boundary) PLUS the caller's extra
    pitch offset. The extra offset is NOT clamped — it may rotate past the
    boundary, in which case the missing region renders black (use
    `black_top_rows` to quantify)."""
    return -applied_elevation(calib, target, elevation_deg) + extra_pitch_deg


def source_quad(calib: CameraCalib,
                target: TargetCamera = DEFAULT_TARGET,
                elevation_deg: float = 0.0,
                extra_pitch_deg: float = 0.0,
                bottom_scale: float = 1.0,
                pin_top: bool = False):
    """Source-image footprint of the virtual view: 4 corner points (TL, TR,
    BR, BL in target order). A rectangle iff no rotation; may extend outside
    the source image when extra_pitch_deg pushes past the data boundary.

    bottom_scale < 1 pulls the bottom edge's corners inward symmetrically
    (keystone correction): the footprint keeps its top edge and vertical
    span but its bottom shortens by the given factor. pin_top snaps the top
    edge exactly onto the source image's top border (corners (0,0) and
    (W_src,0)). Either option turns the warp into a general 4-point
    homography rather than a pure rotation — horizontal scale becomes
    row-dependent while straight lines remain straight."""
    quad = _quad_for_alpha(
        calib, target,
        applied_alpha(calib, target, elevation_deg, extra_pitch_deg))
    if pin_top:
        quad[0] = (0.0, 0.0)
        quad[1] = (float(calib.width), 0.0)
    if abs(bottom_scale - 1.0) > 1e-6:
        (xbr, ybr), (xbl, ybl) = quad[2], quad[3]
        cx_b = (xbr + xbl) / 2.0
        quad[2] = (cx_b + (xbr - cx_b) * bottom_scale, ybr)
        quad[3] = (cx_b + (xbl - cx_b) * bottom_scale, ybl)
    return quad


def _perspective_coeffs_for_quad(quad, target: TargetCamera):
    """PIL PERSPECTIVE coefficients mapping OUTPUT pixels to SOURCE pixels
    for an arbitrary source quad (TL, TR, BR, BL ↔ output corners)."""
    import numpy as np
    dst = [(0.0, 0.0), (float(target.width), 0.0),
           (float(target.width), float(target.height)),
           (0.0, float(target.height))]
    A, b = [], []
    for (X, Y), (x, y) in zip(quad, dst):
        A.append([x, y, 1, 0, 0, 0, -X * x, -X * y]); b.append(X)
        A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y]); b.append(Y)
    return tuple(np.linalg.solve(np.array(A), np.array(b)).tolist())


def black_border_rows(calib: CameraCalib,
                      target: TargetCamera = DEFAULT_TARGET,
                      elevation_deg: float = 0.0,
                      extra_pitch_deg: float = 0.0):
    """(top_rows, bottom_rows) of the output (center column) that map
    outside the source image — the black bands an over-rotation produces.
    (0, 0) when the warp stays inside the data."""
    alpha = applied_alpha(calib, target, elevation_deg, extra_pitch_deg)
    H = _homography_out_to_src(calib, target, alpha)
    u = target.width / 2.0
    top = 0
    for v in range(target.height):
        if _map_point(H, u, v)[1] >= 0.0:
            break
        top += 1
    bottom = 0
    for v in range(target.height - 1, -1, -1):
        if _map_point(H, u, v)[1] <= calib.height:
            break
        bottom += 1
    return top, bottom


def clamp_extra_pitch(calib: CameraCalib,
                      target: TargetCamera = DEFAULT_TARGET,
                      elevation_deg: float = 0.0,
                      desired_extra_deg: float = 0.0) -> float:
    """Largest portion of `desired_extra_deg` (same sign) whose warp
    footprint still stays inside the source image — for callers that want
    extra rotation WITHOUT black borders (e.g. aligning the rear-side views
    down toward the rear-center's axis)."""
    if abs(desired_extra_deg) < _ZERO_ELEVATION_EPS:
        return 0.0

    def ok(extra):
        return _quad_in_bounds(
            _quad_for_alpha(calib, target,
                            applied_alpha(calib, target, elevation_deg, extra)),
            calib)

    if ok(desired_extra_deg):
        return desired_extra_deg
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if ok(desired_extra_deg * mid):
            lo = mid
        else:
            hi = mid
    return desired_extra_deg * lo


def axis_elevation_after(calib: CameraCalib,
                         target: TargetCamera = DEFAULT_TARGET,
                         elevation_deg: float = 0.0,
                         extra_pitch_deg: float = 0.0) -> float:
    """Road-relative elevation of the virtual camera's axis after leveling
    (clamped) and the extra offset. 0 = level; negative = still down."""
    return (elevation_deg
            - applied_elevation(calib, target, elevation_deg)
            + extra_pitch_deg)


def compute_crop_box(calib: CameraCalib,
                     target: TargetCamera = DEFAULT_TARGET,
                     elevation_deg: float = 0.0,
                     extra_pitch_deg: float = 0.0,
                     bottom_scale: float = 1.0,
                     pin_top: bool = False) -> CropBox:
    """Axis-aligned BOUNDING BOX of the source footprint. For elevation ~0
    this is the exact crop rectangle (crop_w = W_t*fx/f_t centered on the
    principal point); for rotated views it bounds the warp trapezoid and is
    reported for logging/UI — the actual mapping is the homography."""
    quad = source_quad(calib, target, elevation_deg, extra_pitch_deg,
                       bottom_scale, pin_top)
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return CropBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))


def min_feasible_focal(calib: CameraCalib,
                       target: TargetCamera = DEFAULT_TARGET,
                       elevation_deg: float = 0.0) -> float:
    """Minimum f_t at which this camera supports the FULL requested pitch
    correction (or, at elevation 0, simply fits the centered crop). Found by
    bisection on f_t with the same in-bounds predicate the warp uses."""
    def feasible(ft: float) -> bool:
        t = TargetCamera(width=target.width, height=target.height, focal=ft)
        return _quad_in_bounds(
            _quad_for_alpha(calib, t, -elevation_deg), calib)

    lo, hi = 1.0, 16000.0
    if not feasible(hi):
        return math.inf
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    return hi


def crop_metadata(calib: CameraCalib,
                  target: TargetCamera = DEFAULT_TARGET,
                  elevation_deg: float = 0.0,
                  extra_pitch_deg: float = 0.0,
                  bottom_scale: float = 1.0,
                  pin_top: bool = False) -> dict:
    """All display-worthy numbers for one camera's crop, for the verifier UI
    and for logging in the batch converter."""
    applied = applied_elevation(calib, target, elevation_deg)
    quad = source_quad(calib, target, elevation_deg, extra_pitch_deg,
                       bottom_scale, pin_top)
    box = compute_crop_box(calib, target, elevation_deg, extra_pitch_deg,
                           bottom_scale, pin_top)
    mf_leveled = min_feasible_focal(calib, target, elevation_deg)
    black_top, black_bottom = (
        black_border_rows(calib, target, elevation_deg, extra_pitch_deg)
        if extra_pitch_deg else (0, 0))
    return {
        'channel': calib.channel,
        'source_size': [calib.width, calib.height],
        'fx': round(calib.fx, 3), 'fy': round(calib.fy, 3),
        'cx': round(calib.cx, 3), 'cy': round(calib.cy, 3),
        'source_hfov_deg': round(calib.hfov_deg, 2),
        'source_vfov_deg': round(calib.vfov_deg, 2),
        'axis_elevation_deg': round(elevation_deg, 2),
        'pitch_correction_applied_deg': round(applied, 2),
        'residual_tilt_deg': round(elevation_deg - applied + extra_pitch_deg, 2),
        'extra_pitch_deg': round(extra_pitch_deg, 2),
        'bottom_scale': round(bottom_scale, 3),
        'pin_top': pin_top,
        'black_top_rows': black_top,
        'black_bottom_rows': black_bottom,
        'axis_elevation_after_deg': round(
            axis_elevation_after(calib, target, elevation_deg,
                                 extra_pitch_deg), 2),
        # Bounding box of the warp footprint (exact crop rect at elevation 0)
        'crop_box': [round(v, 1) for v in box.as_tuple()],
        'crop_size': [round(box.width, 1), round(box.height, 1)],
        'source_quad': [[round(x, 1), round(y, 1)] for x, y in quad],
        'retained_ratio': [round(box.width / calib.width, 3),
                           round(box.height / calib.height, 3)],
        # By construction the warped output has exactly the target camera's
        # FOV (the homography is a pure rotation of the same pinhole).
        'effective_hfov_deg': round(target.hfov_deg, 2),
        'effective_vfov_deg': round(target.vfov_deg, 2),
        # Feasibility of the AUTO-LEVELED geometry (extra pitch excluded —
        # over-rotation is a deliberate choice that black-fills instead).
        'in_bounds': _quad_in_bounds(
            source_quad(calib, target, elevation_deg, 0.0), calib),
        'min_feasible_ft': round(min_feasible_focal(calib, target), 1),
        'min_feasible_ft_leveled': (round(mf_leveled, 1)
                                    if math.isfinite(mf_leveled) else None),
    }


# ---------------------------------------------------------------------------
# Image conversion
# ---------------------------------------------------------------------------

def convert_image(img: Image.Image,
                  calib: CameraCalib,
                  target: TargetCamera = DEFAULT_TARGET,
                  flip: bool = False,
                  elevation_deg: float = 0.0,
                  extra_pitch_deg: float = 0.0,
                  bottom_scale: float = 1.0,
                  pin_top: bool = False) -> Image.Image:
    """Warp one source PIL image into the virtual target camera.

    A TRUE pitch rotation via the rectification homography (PIL PERSPECTIVE
    transform, bicubic): auto-leveling clamped to the source-data boundary,
    plus an optional unclamped `extra_pitch_deg` (+ = further up; regions
    past the data boundary render black). At zero total rotation this
    degenerates to an axis-aligned crop, done as a single subpixel
    crop+resize (LANCZOS). Then an optional horizontal flip for rear-facing
    views; the caller decides `flip` (use `cam_name in REAR_CAMERAS`).
    """
    if (img.width, img.height) != (calib.width, calib.height):
        raise ValueError(
            f'{calib.channel}: image is {img.width}x{img.height} but calib '
            f'says {calib.width}x{calib.height}')
    alpha = applied_alpha(calib, target, elevation_deg, extra_pitch_deg)
    if abs(bottom_scale - 1.0) > 1e-6 or pin_top:
        # Keystone path: arbitrary source quad -> output rect (4-point DLT).
        quad = source_quad(calib, target, elevation_deg, extra_pitch_deg,
                           bottom_scale, pin_top)
        coeffs = _perspective_coeffs_for_quad(quad, target)
        out = img.transform((target.width, target.height), Image.PERSPECTIVE,
                            coeffs, resample=Image.BICUBIC)
        if flip:
            out = out.transpose(Image.FLIP_LEFT_RIGHT)
        return out
    if abs(alpha) < _ZERO_ELEVATION_EPS:
        box = compute_crop_box(calib, target, 0.0)
        if not box.in_bounds(calib.width, calib.height):
            raise ValueError(
                f'{calib.channel}: crop box {box.as_tuple()} exceeds source '
                f'bounds {calib.width}x{calib.height}; minimum feasible '
                f'f_t = {min_feasible_focal(calib, target):.1f}')
        out = img.resize((target.width, target.height), Image.LANCZOS,
                         box=box.as_tuple())
    else:
        H = _homography_out_to_src(calib, target, alpha)
        # PIL PERSPECTIVE data (a..h): maps OUTPUT (x,y) -> SOURCE
        # ((ax+by+c)/(gx+hy+1), (dx+ey+f)/(gx+hy+1)) — normalize H[2][2]=1.
        w = H[2][2]
        coeffs = (H[0][0] / w, H[0][1] / w, H[0][2] / w,
                  H[1][0] / w, H[1][1] / w, H[1][2] / w,
                  H[2][0] / w, H[2][1] / w)
        out = img.transform((target.width, target.height), Image.PERSPECTIVE,
                            coeffs, resample=Image.BICUBIC)
    if flip:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    return out
