from real_world.kinect import KinectClient
from real_world.realsense import RealSense
from real_world.realur5 import UR5
from real_world.wsg50 import WSG50
from real_world.rg2 import RG2

DEFAULT_ORN = [2.22, 2.22, 0.0]

# ---------------------------------------------------------------- reference frames
# TWO DIFFERENT FRAMES MEET IN THIS FORK, AND THEY ARE NOT INTERCHANGEABLE.
#
# 1. CAGE WORLD (GEMSORT). Every grasp point reconstructed from the camera is in this frame:
#    real_world.utils.pix_to_3d_position is given cam_extr = GemsortArmPair.camera_world_transform(), which is a
#    world-from-camera transform built from the MuJoCo cage model. Table top is at +TABLE_Z, z grows upward, and the
#    cage spans roughly x 0..2.8, y 0..2. This is the frame GemsortArmPair.movel plans in.
#
# 2. UPSTREAM SHARED-UR5 FRAME. FlingBot's motion primitives (real_world/fling.py, real_world/stretch.py) were
#    authored for two UR5s standing DIST_UR5 apart, with the origin between them, the table top at -0.15 and the arms
#    separated along y. Those waypoint literals have NOT been converted to the cage frame yet.
#
# The two used to share a single WORKSPACE_SURFACE, which meant the value was silently wrong for one of them. They are
# now separate so that neither can inherit the other's convention by accident.
DIST_UR5 = 1.34
MIN_UR5_BASE_SAFETY_RADIUS = 0.3
UR5_FRAME_WORKSPACE_SURFACE = -0.15   # frame 2 only -- fling.py / stretch.py waypoint literals

# Frame 1: the table top the reconstructed grasp points are measured against.
from gemsort_lib.flingbot_config import TABLE_Z, GRIPPER_OFFSET  # noqa: E402  (kept next to what they define)
WORKSPACE_SURFACE = TABLE_Z

# Lowest a commanded grasp may go. The 18-DOF planner's TCP is the fingertip midpoint, and the gripper's lowest
# geometry sits at essentially the same height (measured: 1 mm below it), so clamping a grasp to the tabletop itself
# drives the gripper 1-6 mm INTO the table once spline overshoot is added. Industrial18DOFPlanner then rejects the
# whole trajectory, because COLLISION_LIMIT_TABLE demands 5 mm of clearance -- and it reports that as
# "REJECTED: unreachable", which reads like an IK failure rather than a grasp that was commanded too low.
# Stand off by the configured grasp clearance, which is well clear of that 5 mm limit.
GRASP_CLEARANCE_ABOVE_SURFACE = GRIPPER_OFFSET

# Upstream accepted any grasp with z <= 0 while the table top sat at -0.15, i.e. up to 15 cm of crumpled cloth above
# the surface. Same allowance, expressed against whatever surface height is configured.
MAX_GRASP_HEIGHT_ABOVE_SURFACE = 0.15

MIN_GRASP_WIDTH = 0.25
MAX_GRASP_WIDTH = 0.6
# workspace pixel crop
WS_PC = [30, -165, 385, -370]
# GEMSORT: annotated on our own cell with `python -m real_world.gemsort_calibrate --workspace`.
try:
    import json as _json, pathlib as _pathlib
    _cal = _json.loads((_pathlib.Path(__file__).with_name('gemsort_calibration.json')).read_text())
    if 'ws_pc' in _cal:
        WS_PC = list(_cal['ws_pc'])
        print(f'[gemsort] workspace crop from gemsort_calibration.json: {WS_PC}')
except Exception:
    pass

UR5_VELOCITY = 0.5
UR5_ACCELERATION = 0.3

CLOTHS_DATASET = {
    'hannes_tshirt': {
        'flatten_area': 0.0524761,
        'cloth_size': (0.45, 0.55),
        'mass': 0.2
    },
}
CURRENT_CLOTH = 'hannes_tshirt'


def get_ur5s():
    return [
        UR5(tcp_ip='XXX.XXX.X.XXX',
            velocity=UR5_VELOCITY,
            acceleration=UR5_ACCELERATION,
            gripper=RG2(tcp_ip='XXX.XXX.X.XXX')),
        UR5(tcp_ip='XXX.XXX.X.XXX',
            velocity=UR5_VELOCITY,
            acceleration=UR5_ACCELERATION,
            gripper=WSG50(tcp_ip='XXX.XXX.X.XXX')),
    ]


def get_top_cam():
    # GEMSORT: our overhead camera is the RealSense on Bernoulli's wrist; the backend and its settings come from
    # real_world/gemsort_calibration.json ("camera"), set from the GUI. Falls back to FlingBot's Kinect client.
    # A configured GEMSORT camera must fail explicitly. Falling back to the
    # upstream placeholder Kinect address hides RealSense profile/USB errors
    # and produces a confusing DNS failure instead of the real cause.
    from real_world.gemsort_camera import make_camera
    return make_camera()


def get_front_cam():
    # The upstream cell used a second RealSense TCP server. GEMSORT currently
    # configures only Bernoulli's wrist D435; do not connect to a placeholder
    # localhost server unless a front camera was explicitly calibrated.
    try:
        import json, pathlib
        cfg = json.loads(pathlib.Path(__file__).with_name('gemsort_calibration.json').read_text()).get('front_camera')
    except Exception:
        cfg = None
    if not cfg:
        return None
    return RealSense(tcp_ip=cfg['ip'], tcp_port=int(cfg.get('port', 12345)),
                     im_h=int(cfg.get('height', 720)), im_w=int(cfg.get('width', 1280)),
                     max_depth=float(cfg.get('max_depth', 3.0)))
