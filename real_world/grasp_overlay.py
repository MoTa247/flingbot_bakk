"""Grasp-verification view: the simulated arms and the real garment under one viewpoint.

At the moment the grippers close there is no direct way to tell whether the fingertips actually sit on the cloth --
the planner believes they do by construction, and the camera has long since stopped looking. This builds one image
that answers it: the MuJoCo model rendered from EXACTLY the pose the wrist RealSense occupied during observation,
with the segmented garment from that observation composited back in.

Because both layers come from the same camera pose, intrinsics and resolution, they share a pixel grid: the render
is produced at the real frame's own width and height, so nothing is ever resized and no aspect ratio can drift. The
only scaling is the optional final downscale for display, which is uniform and applied to the finished composite.
"""
import numpy as np

GARMENT_ALPHA = .65          # keep the gripper visible through the cloth it is supposed to be pinching
CONTOUR_RGB = (40, 255, 40)
COMMANDED_RGB = (255, 80, 80)


def _cv2():
    import cv2
    return cv2


def fovy_degrees(fy, height):
    """Vertical field of view matching a pinhole camera with focal length `fy` over `height` pixels."""
    return float(np.degrees(2.0 * np.arctan(height / (2.0 * float(fy)))))


def project(points_world, world_from_camera, intrinsics):
    """World points -> (u, v) pixels through the real camera's pinhole model. NaN when behind the camera."""
    transform = np.asarray(world_from_camera, float)
    rotation, position = transform[:3, :3], transform[:3, 3]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    out = []
    for point in np.atleast_2d(np.asarray(points_world, float)):
        camera_point = rotation.T @ (point - position)
        if camera_point[2] <= 1e-6:
            out.append((float('nan'), float('nan')))
            continue
        out.append((fx * camera_point[0] / camera_point[2] + cx,
                    fy * camera_point[1] / camera_point[2] + cy))
    return np.asarray(out, float)


def build(sim_rgb, real_rgb, mask=None, commanded_pixels=None, status='', max_width=None):
    """Composite the render and the real garment into one BGR image ready for imshow/imwrite.

    `sim_rgb` and `real_rgb` must already share a shape -- they are produced from the same camera pose at the same
    resolution, so a mismatch means the render was asked for at the wrong size and is a bug, not something to paper
    over by resizing one of them.
    """
    cv2 = _cv2()
    sim = np.asarray(sim_rgb)
    real = np.asarray(real_rgb)
    if sim.shape[:2] != real.shape[:2]:
        raise ValueError(f'render is {sim.shape[1]}x{sim.shape[0]} but the camera frame is '
                         f'{real.shape[1]}x{real.shape[0]}; they must match pixel for pixel')

    frame = np.ascontiguousarray(sim[:, :, ::-1])          # RGB -> BGR
    real_bgr = np.ascontiguousarray(real[:, :, ::-1])

    if mask is not None:
        garment = np.asarray(mask).astype(bool)
        if garment.shape != frame.shape[:2]:
            raise ValueError(f'mask is {garment.shape[1]}x{garment.shape[0]}, frame is '
                             f'{frame.shape[1]}x{frame.shape[0]}')
        blended = frame.copy()
        blended[garment] = (GARMENT_ALPHA * real_bgr[garment]
                            + (1.0 - GARMENT_ALPHA) * frame[garment]).astype(np.uint8)
        frame = blended
        contours, _ = cv2.findContours(garment.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, CONTOUR_RGB[::-1], 2)

    if commanded_pixels is not None:
        for (u, v), name in zip(np.atleast_2d(commanded_pixels), ('L', 'R')):
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            centre = (int(round(u)), int(round(v)))
            cv2.drawMarker(frame, centre, COMMANDED_RGB[::-1], cv2.MARKER_CROSS, 26, 2)
            cv2.circle(frame, centre, 13, COMMANDED_RGB[::-1], 1)
            cv2.putText(frame, name, (centre[0] + 16, centre[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, COMMANDED_RGB[::-1], 2)

    for index, text in enumerate([t for t in (status, 'sim arms + real garment, same camera pose  |  '
                                              'cross = commanded grasp') if t]):
        y = 28 + 24 * index
        cv2.putText(frame, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 3)
        cv2.putText(frame, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)

    if max_width and frame.shape[1] > max_width:
        # Uniform scale on the finished composite: both layers shrink together, so the aspect ratio is preserved.
        scale = max_width / frame.shape[1]
        frame = cv2.resize(frame, (int(round(frame.shape[1] * scale)), int(round(frame.shape[0] * scale))),
                           interpolation=cv2.INTER_AREA)
    return frame
