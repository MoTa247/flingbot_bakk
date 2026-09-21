"""GEMSORT robot backend for FlingBot's real-world stack.

FlingBot's real_world code talks to two UR5s through `real_world.ur5_pair.UR5Pair`. Our cell has two rail-mounted KUKA
iiwa arms (9 DoF each: 2 linear axes + 7 joints) driven through `arcpy.robots.Robot18DOF` and the helpers in
`gemsort_lib` (see ../../move_flingbot.py). This module provides `GemsortArmPair` with the SAME method names UR5Pair
exposes, so nothing else in FlingBot has to change:

    homej / movej / movel / move / close_grippers / open_grippers / out_of_the_way / all_ur5s_reached_target

Hardware layout (2026-09-21): BERNOULLI carries the RealSense (camera arm), PASCAL carries the gripper. Both arms can
grasp; only Bernoulli has the camera, so `go_to_camera_pose()` parks Bernoulli in the overhead observation pose and
keeps Pascal out of the camera's view.

Run standalone (inside the `arc` conda env, from the repo root):
    python -m flingbot_bakk.real_world.gemsort_arm_pair --camera-pose      # move Bernoulli over the table
    python -m flingbot_bakk.real_world.gemsort_arm_pair --home
NOT VERIFIED ON HARDWARE YET — CAMERA_POSE_* below are placeholders that must be taught on the real cell.
"""
import argparse
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Observation pose: Bernoulli holds the RealSense centred over the table looking straight down. The joint values must be
# TAUGHT on the real robot (jog into place, read q, paste here); the world target is what they have to achieve.
CAMERA_TARGET_WORLD = dict(x=1.4, y=1.0, z=2.0, look='down')  # CAM_POS_X/Y/Z in gemsort_lib/flingbot_config.py
CAMERA_POSE_ARM = None      # np.ndarray (7,) Bernoulli joint angles — TEACH ME
CAMERA_POSE_RAIL = None     # (rail_x, rail_y) for Bernoulli — TEACH ME
PARK_POSE_ARM = None        # Pascal joints while the camera observes (out of view) — TEACH ME


class MotionPlanningError(RuntimeError):
    """No collision-free 18-DoF plan for the requested tool poses (arm-arm, table or cell collision, or unreachable)."""


def _pose(pose):
    """FlingBot pose → (xyz, MuJoCo quaternion [w, x, y, z]). Accepts [x,y,z,rx,ry,rz] (UR rotation vector,
    e.g. real_world/setup.py DEFAULT_ORN) or [x,y,z,qw,qx,qy,qz]."""
    from scipy.spatial.transform import Rotation
    pose = np.asarray(pose, float).ravel()
    if pose.size == 7:
        return pose[:3], pose[3:]
    if pose.size != 6:
        raise ValueError(f'pose must have 6 (xyz + rotation vector) or 7 (xyz + wxyz) entries, got {pose.size}')
    x, y, z, w = Rotation.from_rotvec(pose[3:]).as_quat()  # scipy returns [x, y, z, w]
    return pose[:3], np.array([w, x, y, z])


class GemsortArmPair:
    """UR5Pair-compatible facade over FlingBotController (2 × 9 DoF).

    `params` in movej/movel follow FlingBot's convention: a sequence of two entries, one per arm, ordered
    [left, right] exactly like UR5Pair. Poses are 6-D (xyz + axis-angle) for movel and joint vectors for movej.
    """

    def __init__(self, controller=None, grippers=None, move_duration=3.0):
        if controller is None:
            controller, grippers = connect()
        self.bot = controller
        self.grippers = grippers
        self.move_duration = move_duration
        self.left = controller.left
        self.right = controller.right
        self._planner = None      # (Industrial18DOFPlanner, GemSortEnvironment, indices), built on first Cartesian move
        self._last_state = None

    # ---------------------------------------------------------------- UR5Pair API
    def all_ur5s_reached_target(self):
        from gemsort_lib.flingbot_controller import snapshot_robot_state
        state = snapshot_robot_state(self.bot)
        return bool(np.all(np.abs(state.get('velocity', np.zeros(1))) < 1e-3)) if isinstance(state, dict) else True

    def homej(self, blocking=True, **kwargs):
        from gemsort_lib.flingbot_config import DEFAULT_POSE_R, LIN_AXIS_X_R_HOME, LIN_AXIS_X_L_HOME, LIN_AXIS_Y_HOME
        self.bot.move_linear_axes((LIN_AXIS_X_R_HOME, LIN_AXIS_Y_HOME), (LIN_AXIS_X_L_HOME, LIN_AXIS_Y_HOME), self.move_duration)
        self.bot.move_arms_sync(DEFAULT_POSE_R, DEFAULT_POSE_R, self.move_duration)

    def movej(self, params, blocking=True, **kwargs):
        left, right = params
        self.bot.move_arms_sync(np.asarray(right, float), np.asarray(left, float), kwargs.get('duration', self.move_duration))

    def movel(self, params, blocking=True, **kwargs):
        """Straight-line move of both tool frames, planned and collision-checked by the gemsort 18-DoF planner.

        `params` = [left_pose, right_pose]; a pose is [x, y, z, rx, ry, rz] (UR rotation vector, FlingBot's convention)
        or [x, y, z, qw, qx, qy, qz]. Both arms are planned TOGETHER, so arm-arm collisions, the table and the cell are
        all checked before anything is commanded (Industrial18DOFPlanner._validate_final_trajectory); a colliding or
        unreachable target raises MotionPlanningError instead of moving.
        """
        traj = self.plan(params, mode=kwargs.pop('mode', 'cartesian'), label=kwargs.pop('label', 'movel'))
        return self.execute(traj, label='movel')

    # ---------------------------------------------------------------- planning / collision checking
    def plan(self, params, mode='cartesian', label='move', steps=None):
        """Plan both arms to `params` without moving. Raises MotionPlanningError if no collision-free path exists."""
        from gemsort_lib.flingbot_planner import MoveMode
        from gemsort_lib.flingbot_config import PLANNER_STEPS
        planner, env, indices = self.planner()
        (target_l, quat_l), (target_r, quat_r) = (_pose(p) for p in params)
        traj = planner.plan_18dof(target_l, quat_l, target_r, quat_r,
                                  mode=MoveMode.CARTESIAN if mode == 'cartesian' else MoveMode.JOINT,
                                  steps=steps or PLANNER_STEPS, label=label)
        if traj is None or not len(traj['q_r']):
            raise MotionPlanningError(f'{label}: no collision-free plan for left={target_l} right={target_r}')
        self._check_endpoint(traj, (target_l, quat_l), (target_r, quat_r), label)
        return traj

    def _check_endpoint(self, traj, left, right, label, pos_tol=.02, rot_tol=.15):
        """The planner's IK can converge short of the goal and still return a collision-free path; verify the final
        pose before anything is commanded (position tolerance m, orientation tolerance rad)."""
        import mujoco
        from scipy.spatial.transform import Rotation
        planner, env, indices = self.planner()
        model, data = planner.model, planner.data
        backup = data.qpos.copy()
        try:
            data.qpos[indices['left_arm']] = traj['q_l'][-1];data.qpos[indices['right_arm']] = traj['q_r'][-1]
            data.qpos[indices['laxis_left']] = traj['lin_l'][-1];data.qpos[indices['laxis_right']] = traj['lin_r'][-1]
            mujoco.mj_forward(model, data)
            id_l, id_r = planner._get_ee_ids()
            for body, (target, quat), side in ((id_l, left, 'left'), (id_r, right, 'right')):
                rot = Rotation.from_quat(np.roll(np.asarray(quat, float), -1))  # [w,x,y,z] → scipy [x,y,z,w]
                error = planner._get_pose_error(body, np.asarray(target, float), rot)
                pos, ang = float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
                if pos > pos_tol or ang > rot_tol:
                    raise MotionPlanningError(f'{label}: planned path does not reach the {side} target '
                                              f'({pos*1000:.0f} mm, {np.degrees(ang):.0f}° off)')
        finally:
            data.qpos[:] = backup;mujoco.mj_forward(model, data)

    def reachable(self, params, mode='cartesian'):
        """(ok, reason) — plan only, never move. Use for FlingBot's check_action_reachability."""
        try:
            self.plan(params, mode=mode, label='reachability')
            return True, ''
        except MotionPlanningError as error:
            return False, str(error)
        except Exception as error:  # IK/solver failures are also 'not reachable'
            return False, f'planner error: {error!r}'

    def execute(self, traj, label='move'):
        from gemsort_lib.flingbot_controller import (execute_optimized_trajectory, wait_for_robots,
                                                     sync_planning_env_to_robot, snapshot_robot_state)
        planner, env, indices = self.planner()
        finish_times = execute_optimized_trajectory(self.bot, traj)
        wait_for_robots(self.bot, finish_times)
        sync_planning_env_to_robot(env, self.bot, indices)
        self._last_state = snapshot_robot_state(self.bot)
        logger.info('%s done', label)
        return True

    def planner(self):
        """Lazily build the MuJoCo planning model (both arms + gripper + TABLE) and the 18-DoF planner."""
        if self._planner is None:
            from arc.mujoco.gemsort_assembly import GemSortEnvironment
            from gemsort_lib.flingbot_utils import inspect_mujoco_structure
            from gemsort_lib.flingbot_planner import Industrial18DOFPlanner
            env = GemSortEnvironment(model_name='flingbot_real_planner', use_image_server=False)
            env.generate_Pascal_Pernoulli(gripper=True, include_table=True)  # table = collision object
            env.compile_model()
            indices = inspect_mujoco_structure(env)
            self._planner = (Industrial18DOFPlanner(env, indices), env, indices)
        return self._planner

    def move(self, move_type, params, blocking=True, **kwargs):
        return getattr(self, f'move{move_type}')(params, blocking=blocking, **kwargs)

    def check_action_reachability(self, left_pose, right_pose, **kwargs):
        """FlingBot asks this before executing a grasp pair; plans both arms, moves nothing."""
        return self.reachable([left_pose, right_pose], **kwargs)

    def close_grippers(self, blocking=True, **kwargs):
        self.bot.grasp_grippers(side='both', wait=blocking)

    def open_grippers(self, blocking=True, **kwargs):
        self.bot.release_grippers(side='both', wait=blocking)

    def out_of_the_way(self):
        self.homej()

    # ---------------------------------------------------------------- gemsort extras
    def go_to_camera_pose(self):
        """Park Bernoulli so the wrist RealSense looks straight down at the table centre, Pascal out of view."""
        if CAMERA_POSE_ARM is None or CAMERA_POSE_RAIL is None:
            raise RuntimeError('CAMERA_POSE_ARM / CAMERA_POSE_RAIL are not taught yet — jog the cell and fill them in')
        from gemsort_lib.flingbot_config import LIN_AXIS_X_L_HOME, LIN_AXIS_Y_HOME
        self.bot.move_linear_axes(tuple(CAMERA_POSE_RAIL), (LIN_AXIS_X_L_HOME, LIN_AXIS_Y_HOME), self.move_duration)
        park = PARK_POSE_ARM if PARK_POSE_ARM is not None else CAMERA_POSE_ARM
        self.bot.move_arms_sync(np.asarray(CAMERA_POSE_ARM, float), np.asarray(park, float), self.move_duration)


def connect(sim=False):
    """(FlingBotController, DualGripperManager) on the real cell; import-time heavy, hence lazy."""
    from arcpy.robots import Robot18DOF
    from gemsort_lib.flingbot_controller import FlingBotController
    from gemsort_lib.flingbot_gripper import DualGripperManager
    robot = Robot18DOF(None if sim else {})
    controller = FlingBotController(robot)
    grippers = DualGripperManager(None, simulation_left=sim, simulation_right=sim)
    controller.attach_grippers(grippers)
    return controller, grippers


def main():
    p = argparse.ArgumentParser(description='GEMSORT arm pair helper for the FlingBot real-world stack')
    p.add_argument('--camera-pose', action='store_true', help='move Bernoulli into the overhead RealSense pose')
    p.add_argument('--home', action='store_true', help='move both arms to the home pose')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    pair = GemsortArmPair()
    if args.home:
        pair.homej()
    if args.camera_pose:
        pair.go_to_camera_pose()
    logger.info('done')


if __name__ == '__main__':
    main()
