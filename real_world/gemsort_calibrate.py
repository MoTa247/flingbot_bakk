"""Cell calibration for the GEMSORT FlingBot runs: workspace bounding box + Bernoulli's camera observation pose.

Both live in ONE file, `real_world/gemsort_calibration.json`, which `setup.py` (workspace crop) and
`gemsort_arm_pair.py` (camera pose) read at import time. Nothing is hard-coded in the FlingBot sources.

    python -m real_world.gemsort_calibrate --workspace        # drag the box on a live frame, ENTER to accept
    python -m real_world.gemsort_calibrate --save-camera-pose # store the CURRENT robot pose as the observation pose
    python -m real_world.gemsort_calibrate --show             # print what is stored

Workspace: FlingBot crops every observation with WS_PC = [top, bottom, left, right] where bottom/right are negative
offsets from the image edge (real_world/utils.get_workspace_crop). The drag is converted into exactly that form, so the
crop stays valid if the camera resolution changes.
Camera pose: BERNOULLI is the LEFT side of the 18-DoF system (robot_control_gui maps bernoulli → the left arm/rail
endpoints and gripper hand LEFT), PASCAL the right. Jog Bernoulli until the RealSense IMAGE is centred on the table,
then save; Pascal's current joints are kept as the park pose so it stays out of view. Judge the pose by the camera
image, not by the flange: the D435 is mounted ~55 deg off the flange axis, so a vertical flange is not a vertical
camera.
"""
import argparse
import json
from pathlib import Path
import numpy as np

FILE = Path(__file__).with_name('gemsort_calibration.json')
BACKGROUND_FILE = Path(__file__).with_name('gemsort_background.npz')


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
    """Select the workspace on a continuously updating camera preview.

    Drag again at any time to replace the rectangle. Enter/Space accepts the
    current rectangle, r clears it, and c/Escape cancels.
    """
    import cv2
    camera = None
    if frame is None:
        from real_world.setup import get_top_cam
        camera = get_top_cam()

    # Keep the native HighGUI identifier ASCII-only. OpenCV's Qt backend can
    # display Unicode titles, but cvSetMouseCallback fails to look them up and
    # reports a misleading NULL window handler.
    window = 'GEMSORT_workspace_live'
    state = dict(start=None, current=None, rect=None, dragging=False)

    def normalized_rect(start, end):
        x0, y0 = start; x1, y1 = end
        # FlingBot's image transforms and pixel-to-world conversion assume a
        # square workspace. Constrain the drag here instead of failing later
        # in RealWorldEnv.get_obs() with an opaque assertion.
        side = min(abs(x1 - x0), abs(y1 - y0))
        x = x0 if x1 >= x0 else x0 - side
        y = y0 if y1 >= y0 else y0 - side
        return x, y, side, side

    def mouse(event, x, y, _flags, _param):
        point = (int(x), int(y))
        if event == cv2.EVENT_LBUTTONDOWN:
            state.update(start=point, current=point, dragging=True)
        elif event == cv2.EVENT_MOUSEMOVE and state['dragging']:
            state['current'] = point
        elif event == cv2.EVENT_LBUTTONUP and state['dragging']:
            state['current'] = point
            state['rect'] = normalized_rect(state['start'], point)
            state['dragging'] = False

    # Qt's HighGUI backend does not create the native window handle until the
    # first imshow/waitKey. Registering the callback before that raises
    # "NULL window handler" even though namedWindow returned successfully.
    first_rgb = camera.get_rgbd()[0] if camera is not None else frame
    first_image = np.ascontiguousarray(np.asarray(first_rgb)[:, :, ::-1])
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window, first_image)
    cv2.waitKey(250)
    cv2.setMouseCallback(window, mouse)
    print('live camera: position the table/camera, drag the workspace box, then press ENTER/SPACE')
    print('drag again = replace box, r = clear, c/ESC = cancel')
    accepted = False
    last_shape = None
    pending_rgb = first_rgb
    try:
        while True:
            rgb = pending_rgb
            pending_rgb = None
            if rgb is None:
                rgb = camera.get_rgbd()[0] if camera is not None else frame
            image = np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])
            height, width = image.shape[:2]
            last_shape = height, width
            rect = (normalized_rect(state['start'], state['current'])
                    if state['dragging'] else state['rect'])
            if rect is not None:
                x, y, w, h = rect
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, f'{w} x {h} (square)', (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
            cv2.putText(image, 'Drag ROI | ENTER/SPACE save | r clear | c/ESC cancel',
                        (16, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (40, 220, 255), 2)
            cv2.imshow(window, image)
            key = cv2.waitKey(1) & 0xff
            if key in (10, 13, 32):
                if state['rect'] is not None and state['rect'][2] >= 5 and state['rect'][3] >= 5:
                    accepted = True
                    break
                print('draw a box at least 5 x 5 pixels before accepting')
            elif key in (ord('r'), ord('R')):
                state.update(start=None, current=None, rect=None, dragging=False)
            elif key in (27, ord('c'), ord('C')):
                break
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass
        if camera is not None and hasattr(camera, 'close'):
            camera.close()

    if not accepted or state['rect'] is None or last_shape is None:
        print('cancelled');return None
    x, y, w, h = state['rect']
    height, width = last_shape
    if w < 5 or h < 5:
        print('cancelled (box too small)');return None
    top, bottom, left, right = int(y), int(y + h) - height, int(x), int(x + w) - width
    ws_pc = [top, bottom if bottom < 0 else -1, left, right if right < 0 else -1]
    print(f'WS_PC = {ws_pc}  (crop {w} x {h} px from a {width} x {height} frame)')
    save(dict(ws_pc=ws_pc, ws_pc_image_size=[width, height]))
    return ws_pc


def save_camera_pose(pair=None):
    """Store the CURRENT robot state as Bernoulli's (LEFT arm) observation pose and Pascal's (right) park pose."""
    from gemsort_lib.flingbot_controller import snapshot_robot_state
    if pair is None:
        from real_world.gemsort_arm_pair import GemsortArmPair
        pair = GemsortArmPair()
    state = snapshot_robot_state(pair.bot)
    data = dict(camera_pose=dict(arm=np.asarray(state['q_l'], float).tolist(),       # Bernoulli = LEFT
                                 rail=np.asarray(state['lin_l'], float).tolist(),
                                 park_arm=np.asarray(state['q_r'], float).tolist(),  # Pascal = RIGHT, out of view
                                 park_rail=np.asarray(state['lin_r'], float).tolist()))
    save(data)
    print('camera pose (Bernoulli/left):', np.round(data['camera_pose']['arm'], 4),
          'rail', np.round(data['camera_pose']['rail'], 4))
    return data


def import_camera_pose(path, pose_name):
    """Use a 9-DOF pose saved by Manual Robot Control without opening ARC sockets."""
    source = Path(path)
    try:
        poses = json.loads(source.read_text())
        values = poses[pose_name]
    except (OSError, ValueError, KeyError) as error:
        raise RuntimeError(f'cannot load pose {pose_name!r} from {source}: {error}') from error
    if not isinstance(values, list) or len(values) != 9:
        raise ValueError(f'{pose_name!r} must contain rail X/Y plus 7 arm joints; got {len(values)} values')
    values = np.asarray(values, float)
    current = load().get('camera_pose', {})
    camera_pose = dict(current)
    camera_pose.update(rail=values[:2].tolist(), arm=values[2:].tolist(),
                       source_file=str(source.resolve()), source_pose=pose_name)
    save(dict(camera_pose=camera_pose))
    print(f'imported {pose_name!r} from {source.resolve()}')
    print('camera pose (Bernoulli/left):', np.round(camera_pose['arm'], 4),
          'rail', np.round(camera_pose['rail'], 4))
    return camera_pose


def set_camera(kind, serial=None, ip=None, port=None, width=None, height=None):
    from real_world.gemsort_camera import DEFAULT
    config = {**DEFAULT, 'type': kind}
    for key, value in (('serial', serial), ('ip', ip), ('port', port), ('width', width), ('height', height)):
        if value not in (None, ''):
            config[key] = value
    save(dict(camera=config));return config


def capture_background():
    """Capture the empty workspace used by RGB-D cloth segmentation."""
    from real_world.setup import get_top_cam
    camera = get_top_cam()
    try:
        rgb, depth = camera.get_rgbd(repeats=15)
    finally:
        if hasattr(camera, 'close'):
            camera.close()
    np.savez_compressed(BACKGROUND_FILE, rgb=rgb, depth=depth)
    print(f'wrote empty-table RGB-D reference {BACKGROUND_FILE} ({rgb.shape[1]}x{rgb.shape[0]})')
    return BACKGROUND_FILE


def check():
    """Everything the real run needs, with the fix for each missing piece. Returns True when nothing is missing."""
    import importlib.util, shutil
    rows = []
    for module, fix in (('ray', 'pip install ray'), ('torch', 'pip install torch'), ('cv2', 'pip install opencv-python'),
                        ('h5py', 'pip install h5py'), ('filelock', 'pip install filelock'),
                        ('tensorboardX', 'pip install tensorboardX'), ('pyrealsense2', 'pip install pyrealsense2')):
        rows.append((f'python module {module}', importlib.util.find_spec(module) is not None, fix))
    for module, fix in (('arcpy', 'run inside the `arc` conda env'), ('gemsort_lib', 'run from the repo root')):
        rows.append((f'gemsort module {module}', importlib.util.find_spec(module) is not None, fix))
    data = load()
    workspace_ok = False
    workspace_detail = 'GUI: Annotate workspace box (selection must be square)'
    if 'ws_pc' in data and 'ws_pc_image_size' in data:
        try:
            top, bottom, left, right = map(int, data['ws_pc'])
            width, height = map(int, data['ws_pc_image_size'])
            bottom = height + bottom if bottom < 0 else bottom
            right = width + right if right < 0 else right
            crop_width, crop_height = right - left, bottom - top
            workspace_ok = crop_width == crop_height and crop_width > 4
            workspace_detail = (f'GUI: Annotate workspace box (stored crop is '
                                f'{crop_width}x{crop_height}; it must be square)')
        except (TypeError, ValueError):
            pass
    rows.append(('workspace box annotated and square', workspace_ok, workspace_detail))
    rows.append(('camera pose taught', 'camera_pose' in data, 'GUI: Save camera pose (Bernoulli = LEFT)'))
    rows.append(('camera configured', 'camera' in data, 'GUI: camera dropdown'))
    rows.append(('empty-table RGB-D reference', BACKGROUND_FILE.exists(),
                 'clear the workspace, then GUI: Capture empty table'))
    try:  # probe the device directly: importing real_world.* would drag in FlingBot's deps (ray) as well
        import pyrealsense2 as rs
        devices = [(d.get_info(rs.camera_info.serial_number), d.get_info(rs.camera_info.name)) for d in rs.context().devices]
        rows.append((f'RealSense detected ({devices[0][1] if devices else "none connected"})', bool(devices),
                     'plug the wrist camera in / check udev permissions'))
    except Exception as error:
        rows.append((f'RealSense probe failed: {error}', False, 'pip install pyrealsense2'))
    missing = 0
    for name, ok, fix in rows:
        print(f'  [{"OK " if ok else "!! "}] {name}' + ('' if ok else f'   → {fix}'))
        missing += 0 if ok else 1
    print('all good' if not missing else f'{missing} item(s) missing')
    return missing == 0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--workspace', action='store_true', help='annotate the workspace bounding box on a live frame')
    p.add_argument('--save-camera-pose', action='store_true', help='store the current robot pose as the observation pose')
    p.add_argument('--import-camera-pose', nargs=2, metavar=('FILE', 'POSE'),
                   help='import a Manual Robot Control 9-DOF pose without connecting to the robot')
    p.add_argument('--show', action='store_true', help='print the stored calibration')
    p.add_argument('--check', action='store_true', help='check dependencies, calibration and camera')
    p.add_argument('--camera', choices=('realsense_local', 'kinect', 'realsense_tcp'), help='select the top camera')
    p.add_argument('--serial');p.add_argument('--ip');p.add_argument('--port', type=int)
    p.add_argument('--width', type=int);p.add_argument('--height', type=int)
    p.add_argument('--list-cameras', action='store_true', help='list connected RealSense devices')
    p.add_argument('--capture-background', action='store_true', help='capture an empty-table RGB-D segmentation reference')
    args = p.parse_args()
    if args.list_cameras:
        from real_world.gemsort_camera import list_realsense
        for serial, name in list_realsense():
            print(f'  {serial or "-"}  {name}')
    if args.camera:
        print('camera:', set_camera(args.camera, args.serial, args.ip, args.port, args.width, args.height))
    if args.capture_background:
        capture_background()
    if args.check:
        check()
    if args.workspace:
        annotate_workspace()
    if args.save_camera_pose:
        save_camera_pose()
    if args.import_camera_pose:
        import_camera_pose(*args.import_camera_pose)
    if args.show or not (args.workspace or args.save_camera_pose or args.import_camera_pose
                         or args.check or args.camera or args.list_cameras or args.capture_background):
        print(json.dumps(load(), indent=1) if FILE.exists() else f'no calibration yet ({FILE})')


if __name__ == '__main__':
    main()
