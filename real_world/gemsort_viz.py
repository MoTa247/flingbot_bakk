"""Live OpenCV feedback + go/no-go gate for FlingBot runs on the GEMSORT cell.

One window ("GEMSORT FlingBot") showing the real camera image with the predicted grasp pair drawn on top, the value map
(if given) and the current safety levers. Nothing here changes FlingBot's logic; it only draws and, when
GEMSORT_CONFIRM=1, waits for the operator before a move is commanded.

Levers (environment variables, read fresh on every call so the GUI can change them between steps):
    GEMSORT_SPEED_SCALE  0.05 … 1.0   time-scale for every planned trajectory (0.2 = five times slower)
    GEMSORT_CONFIRM      1            wait for 'g' (go) / 'x' (abort) in the window before each commanded move
    GEMSORT_DRY_RUN      1            plan and draw, never command the robot
    GEMSORT_VIZ          0            disable the window entirely (headless)
"""
import os
import numpy as np

WINDOW = 'GEMSORT FlingBot'
_cv2 = None
_failed = False


def levers():
    def number(name, default, lo, hi):
        try:
            return float(np.clip(float(os.environ.get(name, default)), lo, hi))
        except ValueError:
            return default
    return dict(speed=number('GEMSORT_SPEED_SCALE', 1., .05, 1.),
                confirm=os.environ.get('GEMSORT_CONFIRM', '0') == '1',
                dry_run=os.environ.get('GEMSORT_DRY_RUN', '0') == '1',
                enabled=os.environ.get('GEMSORT_VIZ', '1') == '1')


def cv2():
    """Import cv2 lazily; returns None when unavailable or without a display (then we only log)."""
    global _cv2, _failed
    if _cv2 is None and not _failed:
        try:
            import cv2 as module
            if not os.environ.get('DISPLAY') and os.name != 'nt':
                raise RuntimeError('no DISPLAY')
            _cv2 = module
        except Exception as error:  # headless or no OpenCV: degrade to prints
            print(f'[gemsort_viz] window disabled ({error})')
            _failed = True
    return _cv2


def draw(rgb, grasp_pixels=None, grasp_world=None, value_map=None,
         segmentation_mask=None, policy_rgb=None, policy_depth=None,
         status='', extra=None):
    """Return the annotated BGR frame: grasp pair (circles + connecting line), optional value-map heat overlay,
    status line and the active levers. `grasp_pixels` = [(x1, y1), (x2, y2)] in image pixels."""
    module = cv2()
    frame = np.ascontiguousarray(rgb[:, :, ::-1]) if rgb is not None and rgb.ndim == 3 else None
    if module is None:
        return frame
    if frame is None:
        # No camera frame yet: cell staging is planned and confirmed BEFORE the first capture. Draw the status on
        # a blank canvas instead of returning None -- confirm() polls show() for a keypress, so a None frame means
        # no window is ever created and the gate spins forever with nothing on screen to press a key into.
        frame = np.full((320, 720, 3), 28, np.uint8)
    if value_map is not None:
        heat = np.asarray(value_map, np.float32)
        heat = (heat - heat.min()) / max(float(heat.max() - heat.min()), 1e-9)
        heat = module.applyColorMap((heat * 255).astype(np.uint8), module.COLORMAP_INFERNO)
        heat = module.resize(heat, (frame.shape[1], frame.shape[0]), interpolation=module.INTER_NEAREST)
        frame = module.addWeighted(frame, .65, heat, .35, 0)
    if segmentation_mask is not None:
        mask = np.asarray(segmentation_mask).astype(bool)
        if mask.shape != frame.shape[:2]:
            mask = module.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]),
                                 interpolation=module.INTER_NEAREST).astype(bool)
        tint = frame.copy()
        tint[mask] = (40, 210, 40)
        frame = module.addWeighted(frame, .72, tint, .28, 0)
        contours, _ = module.findContours(mask.astype(np.uint8), module.RETR_EXTERNAL,
                                          module.CHAIN_APPROX_SIMPLE)
        module.drawContours(frame, contours, -1, (40, 255, 40), 2)
    if grasp_pixels is not None and len(grasp_pixels) == 2:
        (x1, y1), (x2, y2) = [(int(round(p[0])), int(round(p[1]))) for p in grasp_pixels]
        module.line(frame, (x1, y1), (x2, y2), (60, 220, 60), 2)
        for (x, y), colour, name in (((x1, y1), (255, 160, 0), 'L'), ((x2, y2), (0, 160, 255), 'R')):
            module.circle(frame, (x, y), 9, colour, 2);module.circle(frame, (x, y), 2, colour, -1)
            module.putText(frame, name, (x + 12, y - 8), module.FONT_HERSHEY_SIMPLEX, .6, colour, 2)
    state = levers()
    lines = [status] if status else []
    if grasp_world is not None and len(grasp_world) == 2:
        lines.append('L world %s   R world %s' % tuple(np.round(np.asarray(p, float), 3) for p in grasp_world))
    lines.append('speed %.2f  %s%s' % (state['speed'], 'DRY RUN  ' if state['dry_run'] else '',
                                       'confirm: g = go, x = abort' if state['confirm'] else 'confirm off'))
    lines += list(extra or [])
    for i, text in enumerate(lines):
        y = 24 + 22 * i
        module.putText(frame, text, (12, y), module.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3)
        module.putText(frame, text, (12, y), module.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    if policy_rgb is not None and np.asarray(policy_rgb).ndim == 3:
        # Keep the actual 256x256 policy input unchanged; enlarge only its
        # visualization and place it beside a scaled raw-camera view.
        display_h = min(450, frame.shape[0])
        raw_w = max(1, round(frame.shape[1] * display_h / frame.shape[0]))
        raw_panel = module.resize(frame, (raw_w, display_h), interpolation=module.INTER_AREA)
        policy = np.ascontiguousarray(np.asarray(policy_rgb)[:, :, ::-1])
        policy_panel = module.resize(policy, (display_h, display_h), interpolation=module.INTER_NEAREST)
        module.putText(raw_panel, 'RAW CAMERA + SEGMENTATION', (12, display_h - 16),
                       module.FONT_HERSHEY_SIMPLEX, .58, (0, 0, 0), 3)
        module.putText(raw_panel, 'RAW CAMERA + SEGMENTATION', (12, display_h - 16),
                       module.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 1)
        module.putText(policy_panel, 'POLICY RGB INPUT 256x256', (12, 28),
                       module.FONT_HERSHEY_SIMPLEX, .58, (0, 0, 0), 3)
        module.putText(policy_panel, 'POLICY RGB INPUT 256x256', (12, 28),
                       module.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 1)
        separator = np.full((display_h, 4, 3), 220, np.uint8)
        panels = [raw_panel, separator, policy_panel]
        if policy_depth is not None:
            depth = np.asarray(policy_depth, np.float32)
            valid = np.isfinite(depth) & (depth > 0)
            normalized = np.zeros(depth.shape, np.uint8)
            if np.any(valid):
                lo, hi = np.percentile(depth[valid], (2, 98))
                normalized[valid] = np.clip((depth[valid] - lo) / max(float(hi - lo), 1e-6) * 255,
                                            0, 255).astype(np.uint8)
            depth_panel = module.applyColorMap(normalized, module.COLORMAP_VIRIDIS)
            depth_panel = module.resize(depth_panel, (display_h, display_h),
                                        interpolation=module.INTER_NEAREST)
            module.putText(depth_panel, 'POLICY DEPTH INPUT 256x256', (12, 28),
                           module.FONT_HERSHEY_SIMPLEX, .58, (0, 0, 0), 3)
            module.putText(depth_panel, 'POLICY DEPTH INPUT 256x256', (12, 28),
                           module.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 1)
            panels += [separator.copy(), depth_panel]
        frame = np.concatenate(panels, axis=1)
    return frame


def show(rgb, wait=1, **kwargs):
    """Draw and refresh the window. Returns the pressed key (-1 if none)."""
    module = cv2()
    state = levers()
    if module is None or not state['enabled']:
        if kwargs.get('status'):
            print(f"[gemsort_viz] {kwargs['status']}")
        return -1
    frame = draw(rgb, **kwargs)
    if frame is None:
        return -1
    module.imshow(WINDOW, frame)
    return module.waitKey(max(1, int(wait))) & 0xFF


def confirm(rgb, status='about to move', **kwargs):
    """Go/no-go gate. True = execute. With GEMSORT_CONFIRM=0 it returns True immediately (but still draws)."""
    state = levers()
    if not state['confirm']:
        show(rgb, wait=1, status=status, **kwargs)
        return True
    module = cv2()
    if module is None:
        answer = input(f'[gemsort_viz] {status} — press ENTER to go, or type "x" to abort: ')
        return answer.strip().lower() != 'x'
    while True:
        key = show(rgb, wait=30, status=status + '  [g = go, x = abort]', **kwargs)
        if key in (ord('g'), 13, 32):
            return True
        if key in (ord('x'), 27):
            return False


def close():
    module = cv2()
    if module is not None:
        module.destroyWindow(WINDOW)
