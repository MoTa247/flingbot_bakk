import cv2
import os
import numpy as np
import skimage.morphology as morph
from copy import deepcopy
from pathlib import Path
from real_world.setup import (WORKSPACE_SURFACE, WS_PC, GRASP_CLEARANCE_ABOVE_SURFACE)

BACKGROUND_FILE = Path(__file__).with_name('gemsort_background.npz')
_BACKGROUND = None
_YOLOE = None
_YOLOE_PROMPT = None


def get_largest_component(arr):
    """Return the largest non-empty connected mask without importing PyFlex."""
    labeled, count = morph.label(arr, return_num=True, background=0)
    masks = [(labeled == index).astype(np.uint8)
             for index in range(1, count + 1)]
    masks.sort(key=np.count_nonzero, reverse=True)
    return masks[0] if masks else np.zeros_like(arr, dtype=np.uint8)


class InvalidDepthException(Exception):
    def __init__(self):
        super().__init__('Invalid Depth Point')


def bound_grasp_pos(pos, z_offset=0.05):
    pos = deepcopy(pos)
    # grasp slightly lower than detected depth
    pos[2] -= z_offset
    # ...but never below the height at which the gripper itself would hit the table (see setup.py).
    pos[2] = max(WORKSPACE_SURFACE + GRASP_CLEARANCE_ABOVE_SURFACE, pos[2])
    pos[2] = min(WORKSPACE_SURFACE + 0.1, pos[2])
    return pos


def get_workspace_crop(img):
    retval = img[WS_PC[0]:WS_PC[1], WS_PC[2]:WS_PC[3], ...]
    if retval.ndim < 2 or min(retval.shape[:2]) == 0:
        raise ValueError(f'invalid workspace crop {WS_PC} for image shape {img.shape}')
    if retval.shape[0] != retval.shape[1]:
        raise ValueError(f'FlingBot workspace must be square, but crop {WS_PC} produces '
                         f'{retval.shape[1]}x{retval.shape[0]} pixels; use Annotate workspace again')
    return retval


def _workspace_only(mask):
    mask[:WS_PC[0], ...] = 0
    mask[WS_PC[1]:, ...] = 0
    mask[:, :WS_PC[2]] = 0
    mask[:, WS_PC[3]:] = 0
    return mask


def _background_mask(rgb, depth):
    global _BACKGROUND
    if _BACKGROUND is None:
        if not BACKGROUND_FILE.exists():
            return None
        with np.load(BACKGROUND_FILE) as data:
            _BACKGROUND = (data['rgb'], data['depth'])
    bg_rgb, bg_depth = _BACKGROUND
    if bg_rgb.shape != rgb.shape or bg_depth.shape != depth.shape:
        raise ValueError(f'empty-table reference is {bg_rgb.shape[:2]}, current frame is {rgb.shape[:2]}; '
                         'capture the background again')
    current_lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.int16)
    background_lab = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2LAB).astype(np.int16)
    colour_delta = np.linalg.norm(current_lab - background_lab, axis=2)
    valid_depth = (depth > 0) & (bg_depth > 0)
    # Cloth is closer to the wrist camera than the empty table. Colour also
    # catches very flat fabric whose depth difference approaches D435 noise.
    depth_foreground = valid_depth & ((bg_depth - depth) > .007)
    mask = ((colour_delta > 24.) | depth_foreground).astype(np.uint8)
    mask = _workspace_only(mask)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return get_largest_component(mask).astype(np.uint8)


def _yoloe_mask(rgb):
    """Prompted YOLOE instance segmentation; model and text embedding are cached."""
    global _YOLOE, _YOLOE_PROMPT
    try:
        from ultralytics import YOLOE
    except ImportError as error:
        raise RuntimeError('YOLOE segmentation selected but ultralytics is not installed; click '
                           '"Install / update FlingBot env"') from error
    prompt = os.environ.get('GEMSORT_SEGMENT_PROMPT', 'cloth garment').strip()
    if not prompt:
        raise ValueError('YOLOE segmentation prompt is empty')
    if _YOLOE is None:
        model_name = os.environ.get('GEMSORT_YOLOE_MODEL', 'yoloe-11s-seg.pt')
        print(f'[gemsort] loading YOLOE segmenter {model_name!r}')
        _YOLOE = YOLOE(model_name)
    if _YOLOE_PROMPT != prompt:
        print(f'[gemsort] YOLOE text prompt: {prompt!r}')
        _YOLOE.set_classes([prompt])
        _YOLOE_PROMPT = prompt
    # Ultralytics ndarray sources are BGR; the camera API is RGB.
    result = _YOLOE.predict(rgb[:, :, ::-1], verbose=False,
                            conf=float(os.environ.get('GEMSORT_SEGMENT_CONFIDENCE', '.15')))[0]
    if result.masks is None or not len(result.masks.data):
        return np.zeros(rgb.shape[:2], np.uint8)
    masks = result.masks.data.detach().cpu().numpy()
    mask = np.any(masks > .5, axis=0).astype(np.uint8)
    if mask.shape != rgb.shape[:2]:
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = _workspace_only(mask)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = get_largest_component(mask).astype(np.uint8)
    try:
        dilation = max(0, min(100, int(os.environ.get('GEMSORT_MASK_DILATION_PX', '12'))))
    except ValueError:
        dilation = 12
    if dilation:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = _workspace_only(mask)
    return mask.astype(np.uint8)


def get_cloth_mask(rgb, depth=None):
    backend = os.environ.get('GEMSORT_SEGMENTER', 'rgbd_background').strip().lower()
    if backend == 'yoloe':
        return _yoloe_mask(rgb)
    if backend not in ('rgbd_background', 'rgbd', 'hsv'):
        raise ValueError(f'unknown GEMSORT_SEGMENTER {backend!r} (yoloe | rgbd_background | hsv)')
    if backend != 'hsv' and depth is not None:
        mask = _background_mask(rgb, depth)
        if mask is not None:
            return mask
    h, w, c = rgb.shape
    if h == 720 and w == 1280:
        rgb[:WS_PC[0], ...] = 0
        rgb[WS_PC[1]:, ...] = 0
        rgb[:, :WS_PC[2], :] = 0
        rgb[:, WS_PC[3]:, :] = 0
    """
    Segments out black backgrounds
    """
    bottom = (0, 0, 0)
    top = (255, 255, 125)
    mask = cv2.inRange(cv2.cvtColor(
        rgb, cv2.COLOR_RGB2HSV), bottom, top)
    mask = (mask == 0).astype(np.uint8)
    if mask.shape[0] != mask.shape[1]:
        mask[:, :int(mask.shape[1]*0.2)] = 0
        mask[:, -int(mask.shape[1]*0.2):] = 0
    return get_largest_component(mask).astype(np.uint8)


def compute_coverage(rgb, depth=None):
    mask = get_cloth_mask(rgb=rgb, depth=depth)
    return np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])


def pix_to_3d_position(
        x, y, depth_image, cam_intr, cam_extr, cam_depth_scale):
    # Get click point in camera coordinates
    click_z = depth_image[y, x] * cam_depth_scale
    click_x = (x-cam_intr[0, 2]) * \
        click_z/cam_intr[0, 0]
    click_y = (y-cam_intr[1, 2]) * \
        click_z/cam_intr[1, 1]
    if click_z == 0:
        raise InvalidDepthException
    click_point = np.asarray([click_x, click_y, click_z])
    click_point = np.append(click_point, 1.0).reshape(4, 1)

    # Convert camera coordinates to robot coordinates
    target_position = np.dot(cam_extr, click_point)
    target_position = target_position[0:3, 0]
    return target_position


def pick_place_primitive_helper(ur5, pick_pose, place_pose,
                                backup=0.02, **kwargs):
    ur5.gripper.open(blocking=True)
    pick_pose = deepcopy(pick_pose)
    if not ur5.movej(
            params=pick_pose, blocking=True,
            use_pos=True, **kwargs):
        return False
    ur5.gripper.close(blocking=True)
    post_grasp_pose = deepcopy(pick_pose)
    post_grasp_pose[2] += backup
    post_grasp_kwargs = deepcopy(kwargs)
    post_grasp_kwargs['j_vel'] = 0.01
    post_grasp_kwargs['j_acc'] = 0.01
    if not ur5.movel(params=post_grasp_pose, blocking=True, use_pos=True,
                     **post_grasp_kwargs):
        return False
    if not ur5.movej(
            params=place_pose, blocking=True,
            use_pos=True, **kwargs):
        return False
    ur5.gripper.open(blocking=True)
    return True
