"""Cell calibration for the GEMSORT FlingBot runs: workspace bounding box + Bernoulli's camera observation pose.

Both live in ONE file, `real_world/gemsort_calibration.json`, which `setup.py` (workspace crop) and
`gemsort_arm_pair.py` (camera pose) read at import time. Nothing is hard-coded in the FlingBot sources.

    python -m real_world.gemsort_calibrate --workspace        # drag the box on a live frame, ENTER to accept
    python -m real_world.gemsort_calibrate --save-camera-pose # store the CURRENT robot pose as the observation pose
    python -m real_world.gemsort_calibrate --show             # print what is stored

Workspace: FlingBot crops every observation with WS_PC = [top, bottom, left, right] where bottom/right are negative
offsets from the image edge (real_world/utils.get_workspace_crop). The drag is converted into exactly that form, so the
crop stays valid if the camera resolution changes.
Camera pose: BERNOULLI is the RIGHT side of the 18-DoF system (robot_control_gui maps bernoulli → RIGHT), PASCAL the
left. Jog Bernoulli until the RealSense looks straight down at the table, then save; Pascal's current joints are kept
as the park pose so it stays out of view.
"""
import argparse
import json
from pathlib import Path
import numpy as np

FILE = Path(__file__).with_name('gemsort_calibration.json')


def load():
    try:
        return json.loads(FILE.read_text())
    except (OSError, ValueError):
        return {}


def save(data):
    current = load();current.update(data)
    FILE.write_text(json.dumps(current, indent=1));print(f'wrote {FILE}')
    return current


def annotate_workspace(frame=None):
    """Drag the workspace box on a camera frame; stores WS_PC in FlingBot's [top, bottom, left, right] convention."""
    import cv2
    if frame is None:
        from real_world.setup import get_top_cam
        frame = get_top_cam().get_rgbd()[0]
    image = np.ascontiguousarray(np.asarray(frame)[:, :, ::-1])
    height, width = image.shape[:2]
    print(f'camera frame {width} x {height}: drag the workspace box, ENTER/SPACE to accept, c to cancel')
    x, y, w, h = cv2.selectROI('GEMSORT workspace', image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow('GEMSORT workspace')
    if w < 5 or h < 5:
        print('cancelled (box too small)');return None
    top, bottom, left, right = int(y), int(y + h) - height, int(x), int(x + w) - width
    ws_pc = [top, bottom if bottom < 0 else -1, left, right if right < 0 else -1]
    print(f'WS_PC = {ws_pc}  (crop {w} x {h} px from a {width} x {height} frame)')
    save(dict(ws_pc=ws_pc, ws_pc_image_size=[width, height]))
    return ws_pc


def save_camera_pose(pair=None):
    """Store the CURRENT robot state as Bernoulli's (right arm) observation pose and Pascal's park pose."""
    from gemsort_lib.flingbot_controller import snapshot_robot_state
    if pair is None:
        from real_world.gemsort_arm_pair import GemsortArmPair
        pair = GemsortArmPair()
    state = snapshot_robot_state(pair.bot)
    data = dict(camera_pose=dict(arm=np.asarray(state['q_r'], float).tolist(),      # Bernoulli = RIGHT
                                 rail=np.asarray(state['lin_r'], float).tolist(),
                                 park_arm=np.asarray(state['q_l'], float).tolist(),  # Pascal = LEFT, out of view
                                 park_rail=np.asarray(state['lin_l'], float).tolist()))
    save(data)
    print('camera pose (Bernoulli/right):', np.round(data['camera_pose']['arm'], 4),
          'rail', np.round(data['camera_pose']['rail'], 4))
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--workspace', action='store_true', help='annotate the workspace bounding box on a live frame')
    p.add_argument('--save-camera-pose', action='store_true', help='store the current robot pose as the observation pose')
    p.add_argument('--show', action='store_true', help='print the stored calibration')
    args = p.parse_args()
    if args.workspace:
        annotate_workspace()
    if args.save_camera_pose:
        save_camera_pose()
    if args.show or not (args.workspace or args.save_camera_pose):
        print(json.dumps(load(), indent=1) if FILE.exists() else f'no calibration yet ({FILE})')


if __name__ == '__main__':
    main()
