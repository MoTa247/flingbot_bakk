"""GEMSORT robot backend for FlingBot's real-world stack.

FlingBot's real_world code talks to two UR5s through `real_world.ur5_pair.UR5Pair`. Our cell has two rail-mounted KUKA
iiwa arms (9 DoF each: 2 linear axes + 7 joints) driven through `arcpy.robots.Robot18DOF` and the helpers in
`gemsort_lib` (see ../../move_flingbot.py). This module provides `GemsortArmPair` with the SAME method names UR5Pair
exposes, so nothing else in FlingBot has to change:

    homej / movej / movel / move / close_grippers / open_grippers / out_of_the_way / all_ur5s_reached_target

Hardware layout (2026-09-21): BERNOULLI carries the RealSense (camera arm), PASCAL carries the gripper. Both arms can
grasp; only Bernoulli has the camera, so `go_to_camera_pose()` parks Bernoulli in the overhead observation pose and
keeps Pascal out of the camera's view.

SIDES -- BERNOULLI IS THE *LEFT* SIDE OF THE 18-DOF SYSTEM, PASCAL IS THE RIGHT.
robot_control_gui.parse_args maps bernoulli to the left arm/rail endpoints (SIM_PORT_IIWA_L 10000/11000 +
SIM_PORT_RAIL_L 30002/31002, real subnet 192.168.1.x = REAL_*_LEFT, gripper hand LEFT) and pascal to the right ones.
In the MuJoCo model arc.mujoco.gemsort_assembly attaches the LEFT iiwa first, so the left arm is `cage/iiwa` and the
right arm is `cage/iiwa_1` (gemsort_lib.flingbot_utils.inspect_mujoco_structure follows the same rule). Anything that
touches the camera therefore has to use the LEFT indices and `cage/iiwa`; the right side is Pascal's.

Run standalone (inside the `arc` conda env, from the repo root):
    python -m flingbot_bakk.real_world.gemsort_arm_pair --camera-pose      # move Bernoulli over the table
    python -m flingbot_bakk.real_world.gemsort_arm_pair --home
NOT VERIFIED ON HARDWARE YET — CAMERA_POSE_* below are placeholders that must be taught on the real cell.
"""
import argparse
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

# Observation pose: Bernoulli holds the RealSense centred over the table looking straight down. The joint values must be
# TAUGHT on the real robot (jog into place, read q, paste here); the world target is what they have to achieve.
# CAMERA_TARGET_WORLD is where the *camera optical frame* has to end up, NOT the flange: the D435 hand-eye
# (gemsort_demo_tool/.../wrist_camera_extrinsics.json) is tilted -54.7 deg about the flange x axis, so a flange that
# points straight down leaves the camera looking ~55 deg off vertical. Teach the pose against the camera image.
CAMERA_TARGET_WORLD = dict(x=1.4, y=1.0, z=2.0, look='down')  # CAM_POS_X/Y/Z in gemsort_lib/flingbot_config.py
CAMERA_POSE_ARM = None      # Bernoulli (LEFT) joint angles — taught with gemsort_calibrate --save-camera-pose
CAMERA_POSE_RAIL = None     # (rail_x, rail_y) for Bernoulli, in LEFT-rail local coordinates
PARK_POSE_ARM = None        # Pascal (RIGHT) joints while the camera observes (out of view)
PARK_POSE_RAIL = None


def _load_camera_pose():
    """Read the taught observation pose from real_world/gemsort_calibration.json (written by gemsort_calibrate)."""
    global CAMERA_POSE_ARM, CAMERA_POSE_RAIL, PARK_POSE_ARM, PARK_POSE_RAIL
    import json, pathlib
    try:
        pose = json.loads(pathlib.Path(__file__).with_name('gemsort_calibration.json').read_text())['camera_pose']
    except Exception:
        return False
    CAMERA_POSE_ARM = np.asarray(pose['arm'], float);CAMERA_POSE_RAIL = np.asarray(pose['rail'], float)
    PARK_POSE_ARM = np.asarray(pose.get('park_arm'), float) if pose.get('park_arm') is not None else None
    PARK_POSE_RAIL = np.asarray(pose.get('park_rail'), float) if pose.get('park_rail') is not None else None
    return True


_load_camera_pose()


def _check_within_joint_range(model, qpos_addresses, values, label):
    """Fail loudly when a stored pose does not fit the axis it is being written to.

    mj_forward does NOT clamp qpos, so feeding e.g. a left-rail reading (travel 1.660..2.580) into the right rail
    (travel 0.410..1.260) silently yields forward kinematics for a carriage position that cannot physically exist --
    which is exactly how a pose taught on the wrong arm turns into a plausible-looking but wrong camera transform.
    """
    import mujoco
    addresses = np.atleast_1d(np.asarray(qpos_addresses, int))
    values = np.atleast_1d(np.asarray(values, float))
    if addresses.size != values.size:
        raise ValueError(f'{label}: expected {addresses.size} values, got {values.size}')
    for address, value in zip(addresses, values):
        matches = np.flatnonzero(model.jnt_qposadr == address)
        if not matches.size:
            continue
        joint = int(matches[0])
        if not model.jnt_limited[joint]:
            continue
        low, high = (float(bound) for bound in model.jnt_range[joint])
        if not low - 1e-6 <= value <= high + 1e-6:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            raise ValueError(f'{label}: {name} = {value:.4f} is outside its travel [{low:.4f}, {high:.4f}]. '
                             f'The stored pose does not belong to this axis -- re-teach it with '
                             f'`gemsort_calibrate --save-camera-pose` on Bernoulli (LEFT).')


def _warn_if_not_overhead(world_from_camera):
    """Sanity-check the observation pose the whole pixel-to-world chain hangs off.

    An overhead pose has the camera above the table top looking down at it. When the stored joints are not the pose
    that was actually taught, the camera typically ends up low and tilted, and every reconstructed grasp then lands
    metres away along the optical axis instead of on the table -- with no other symptom than implausible coordinates.
    Warn loudly here so the cause is named at startup; RealWorldEnv.check_action still rejects the resulting points.
    """
    from gemsort_lib.flingbot_config import TABLE_Z
    position = np.asarray(world_from_camera, float)[:3, 3]
    view = np.asarray(world_from_camera, float)[:3, 2]   # camera optical +Z, i.e. where it looks
    down_deg = float(np.degrees(np.arccos(np.clip(-view[2], -1.0, 1.0))))
    problems = []
    if position[2] <= TABLE_Z:
        problems.append(f'camera is at z={position[2]:.3f} m, at or below the {TABLE_Z:.3f} m table top')
    if down_deg > 45.0:
        problems.append(f'camera looks {down_deg:.0f} deg off straight-down')
    if problems:
        logger.warning('observation pose does not look overhead (%s). Camera world position %s, optical axis %s. '
                       'Re-teach it: jog Bernoulli (LEFT) until the RealSense IMAGE is centred on the table, then '
                       '"Use saved Manual pose" / `gemsort_calibrate --save-camera-pose`. Remember the D435 sits '
                       '~55 deg off the flange axis, so a vertical flange is NOT a vertical camera.',
                       '; '.join(problems), np.round(position, 3), np.round(view, 3))
    return not problems


class MotionPlanningError(RuntimeError):
    """No collision-free 18-DoF plan for the requested tool poses (arm-arm, table or cell collision, or unreachable)."""


def _warn_if_start_state_unusable(env):
    """Say so when the seeded start state is one no plan can succeed from.

    A start state that is already in collision, or whose gripper hangs below the tabletop, makes every Cartesian plan
    sweep through the table on its way out -- and the planner reports that as "REJECTED: unreachable", which reads
    like an IK or reachability problem rather than a bad start. Naming it here keeps the next hour off a false trail.
    """
    import mujoco
    from gemsort_lib.flingbot_config import TABLE_Z, GRIPPER_LENGTH
    model, data = env.model, env.data
    penetrating = [i for i in range(data.ncon) if data.contact[i].dist < 0]
    if penetrating:
        contact = data.contact[penetrating[0]]
        logger.warning('seeded start state is already in collision (%d penetrating contacts, e.g. %s <-> %s at '
                       '%.1f mm). Every plan plan from it will be rejected.', len(penetrating),
                       mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1),
                       mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2), contact.dist * 1000)

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

    def __init__(self, controller=None, grippers=None, move_duration=3.0, sim=None,
                 simulation_left=None, simulation_right=None):
        if controller is None:
            simulation_left, simulation_right = simulation_modes(
                sim=sim, simulation_left=simulation_left, simulation_right=simulation_right)
            controller, grippers = connect(
                simulation_left=simulation_left, simulation_right=simulation_right)
        self.bot = controller
        self.grippers = grippers
        self.move_duration = move_duration
        self.left = controller.left
        self.right = controller.right
        self._planner = None      # (Industrial18DOFPlanner, GemSortEnvironment, indices), built on first Cartesian move
        self._hover_config = None # joint solution for hover-ready, solved once by hover_configuration()
        self._renderer = None     # ((w, h), mujoco.Renderer) cached: building a GL context per frame is slow
        self._target = None       # last commanded {q_l, q_r, lin_l, lin_r}, for all_ur5s_reached_target
        self.preview = None       # latest camera frame, set by RealWorldEnv for the confirmation window

    # ---------------------------------------------------------------- UR5Pair API
    @staticmethod
    def _dry_run():
        """Return the process-wide safety state without importing GUI helpers."""
        return os.environ.get('GEMSORT_DRY_RUN', '1') == '1'

    def all_ur5s_reached_target(self, tol=.01, rail_tol=.005):
        """True when both arms and both rails are within tolerance of the last commanded target.
        (snapshot_robot_state reports q/rail positions, not velocities, so we compare against the command.)"""
        from gemsort_lib.flingbot_controller import snapshot_robot_state
        if self._target is None:
            return True
        now = snapshot_robot_state(self.bot);want = self._target
        return (np.max(np.abs(now['q_l'] - want['q_l'])) < tol and np.max(np.abs(now['q_r'] - want['q_r'])) < tol
                and np.max(np.abs(now['lin_l'] - want['lin_l'])) < rail_tol
                and np.max(np.abs(now['lin_r'] - want['lin_r'])) < rail_tol)

    def homej(self, blocking=True, **kwargs):
        if self._dry_run():
            logger.warning('homej skipped (GEMSORT_DRY_RUN=1)')
            return False
        raise MotionPlanningError('homej disabled: the legacy direct joint move is not a collision-validated '
                                  '18-DOF trajectory')

    def movej(self, params, blocking=True, **kwargs):
        if self._dry_run():
            logger.warning('movej skipped (GEMSORT_DRY_RUN=1)')
            return False
        raise MotionPlanningError('movej disabled: a 7-DoF UR5 joint target does not define the GEMSORT rail '
                                  'positions and cannot be safely executed')

    def movel(self, params, blocking=True, **kwargs):
        """Straight-line move of both tool frames, planned and collision-checked by the gemsort 18-DoF planner.

        `params` = [left_pose, right_pose]; a pose is [x, y, z, rx, ry, rz] (UR rotation vector, FlingBot's convention)
        or [x, y, z, qw, qx, qy, qz]. Both arms are planned TOGETHER, so arm-arm collisions, the table and the cell are
        all checked before anything is commanded (Industrial18DOFPlanner._validate_final_trajectory); a colliding or
        unreachable target raises MotionPlanningError instead of moving.
        """
        preview = kwargs.pop('preview', None) if 'preview' in kwargs else getattr(self, 'preview', None)
        # Carry the caller's label through to execute() as well: it is what the confirm window shows, and an
        # operator gating every move needs to see "descent" rather than "movel" for each of them.
        label = kwargs.pop('label', 'movel')
        traj = self.plan(params, mode=kwargs.pop('mode', 'cartesian'), label=label)
        return self.execute(traj, label=label, preview=preview)

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

    def execute(self, traj, label='move', preview=None):
        """Command a planned trajectory, after applying the safety levers (gemsort_viz.levers):
        GEMSORT_SPEED_SCALE stretches the trajectory timing, GEMSORT_DRY_RUN skips execution entirely and
        GEMSORT_CONFIRM waits for the operator's go in the OpenCV window."""
        from gemsort_lib.flingbot_controller import (execute_optimized_trajectory, wait_for_robots,
                                                     sync_planning_env_to_robot, snapshot_robot_state)
        from real_world import gemsort_viz
        planner, env, indices = self.planner()
        state = gemsort_viz.levers()
        traj = self._scale_speed(traj, state['speed'])
        seconds = float(np.asarray(traj['times'])[-1] - np.asarray(traj['times'])[0])
        status = f"{label}: {seconds:.1f} s at speed {state['speed']:.2f}"
        if not gemsort_viz.confirm(preview, status=status):
            logger.warning('%s aborted by operator', label)
            return False
        if state['dry_run']:
            # Advance the planning model to where this trajectory would have ended. Without it every dry-run step
            # re-plans from the cell's rest state, so a multi-step sequence (stage -> observe -> back to hover)
            # can never be rehearsed: step 2 would be planned from step 0. Nothing is commanded either way.
            from gemsort_lib.flingbot_controller import apply_state_to_planning_env
            apply_state_to_planning_env(env, indices, {
                'q_l': np.asarray(traj['q_l'][-1], float), 'q_r': np.asarray(traj['q_r'][-1], float),
                'lin_l': np.asarray(traj['lin_l'][-1], float), 'lin_r': np.asarray(traj['lin_r'][-1], float)})
            logger.warning('%s NOT executed (GEMSORT_DRY_RUN=1); plan was collision-checked and would take %.1f s',
                           label, seconds)
            return True
        finish_times = execute_optimized_trajectory(self.bot, traj)
        wait_for_robots(self.bot, finish_times)
        sync_planning_env_to_robot(env, self.bot, indices)
        self._target = dict(q_l=np.asarray(traj['q_l'][-1], float), q_r=np.asarray(traj['q_r'][-1], float),
                            lin_l=np.asarray(traj['lin_l'][-1], float), lin_r=np.asarray(traj['lin_r'][-1], float))
        logger.info('%s done', label)
        return True

    @staticmethod
    def _scale_speed(traj, scale):
        """Stretch the planned timing (scale 0.2 ⇒ five times slower). Joint/rail waypoints are unchanged, so the
        path — and therefore its collision clearance — stays exactly as validated."""
        if scale >= .999 or 'times' not in traj:
            return traj
        out = dict(traj);out['times'] = (np.asarray(traj['times'], float) / max(scale, 1e-3)).tolist()
        for key in ('qd_r', 'qd_l', 'lind_r', 'lind_l'):  # velocities, if the generator provided them
            if key in traj and traj[key] is not None:
                out[key] = (np.asarray(traj[key], float) * scale).tolist()
        return out

    # ---------------------------------------------------------------- cell startup / hover cycle
    def joint_state(self):
        """Current 18-DOF state as {q_l, q_r, lin_l, lin_r}.

        Reads the robot when there is one; in dry run there are no sockets, so report what the planning model was
        seeded with. Either way the caller gets the state that planning will actually start from.
        """
        if not self._dry_run() and not isinstance(self.bot, _DryRunController):
            from gemsort_lib.flingbot_controller import snapshot_robot_state
            return snapshot_robot_state(self.bot)
        _planner, env, indices = self.planner()
        return {'q_l': env.data.qpos[indices['left_arm']].copy(),
                'q_r': env.data.qpos[indices['right_arm']].copy(),
                'lin_l': env.data.qpos[indices['laxis_left']].copy(),
                'lin_r': env.data.qpos[indices['laxis_right']].copy()}

    def where_are_the_arms(self, arm_tol=.15, rail_tol=.05, tcp_tol=.05):
        """Classify the current configuration: 'hanging', 'hover', 'observation' or 'elsewhere'.

        HANGER_POSE is all-zeros, so "hanging" is the arms hanging straight down off the gantry -- the state the
        cell powers up in and the one the operator leaves it in. It is NOT a state any Cartesian move can start
        from safely once the table is in the way, which is why the run has to stage out of it first.
        """
        import mujoco
        from gemsort_lib.flingbot_config import hover_tcp_targets, GRIPPER_LENGTH
        state = self.joint_state()
        if max(np.max(np.abs(state['q_l'])), np.max(np.abs(state['q_r']))) < arm_tol:
            return 'hanging', state

        _load_camera_pose()
        if (CAMERA_POSE_ARM is not None
                and np.max(np.abs(state['q_l'] - np.asarray(CAMERA_POSE_ARM, float))) < arm_tol
                and np.max(np.abs(state['lin_l'] - np.asarray(CAMERA_POSE_RAIL, float))) < rail_tol):
            return 'observation', state

        planner, env, indices = self.planner()
        backup = env.data.qpos.copy()
        try:
            env.data.qpos[indices['left_arm']] = state['q_l'];   env.data.qpos[indices['right_arm']] = state['q_r']
            env.data.qpos[indices['laxis_left']] = state['lin_l']; env.data.qpos[indices['laxis_right']] = state['lin_r']
            mujoco.mj_forward(env.model, env.data)
            want_l, want_r = hover_tcp_targets()
            reached = []
            for body_name, want in (('cage/iiwa/iiwa_link_7', want_l), ('cage/iiwa_1/iiwa_link_7', want_r)):
                body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                tcp = env.data.xpos[body] + env.data.xmat[body].reshape(3, 3) @ np.array([0, 0, GRIPPER_LENGTH])
                reached.append(float(np.linalg.norm(tcp - want)) < tcp_tol)
        finally:
            env.data.qpos[:] = backup
            mujoco.mj_forward(env.model, env.data)
        return ('hover' if all(reached) else 'elsewhere'), state

    def plan_ptp(self, q_l, q_r, lin_l, lin_r, label='ptp'):
        """Collision-checked joint+rail point-to-point. Moves nothing."""
        planner, _env, _indices = self.planner()
        traj = planner.plan_simple_ptp(np.asarray(q_l, float), np.asarray(q_r, float),
                                       np.asarray(lin_l, float), np.asarray(lin_r, float), label=label)
        if traj is None:
            raise MotionPlanningError(f'{label}: no collision-free joint-space path from the current state')
        return traj

    def render_from_camera(self, world_from_camera, fovy_deg, width, height):
        """Render the planning model from an arbitrary world camera pose, as HxWx3 uint8 RGB.

        Used to put the simulated arms under the same viewpoint the wrist RealSense had, so a real frame and the
        kinematic model can be compared pixel for pixel. Three MuJoCo details matter and are handled here:

        * the offscreen framebuffer defaults to 640x480 and Renderer refuses anything larger, so it is widened to
          the requested size -- rendering small and scaling up would throw away exactly the precision this view
          exists to show;
        * `ipd` defaults to 68 mm and the scene cameras are the two stereo eyes, which lands the render half an
          interpupillary distance (34 mm) away from the pose asked for. Zeroed;
        * the free camera is parameterised by azimuth/elevation, which is degenerate at nadir. Elevation comes
          from the forward vector but azimuth is taken from the camera's UP vector, which stays well conditioned
          when looking straight down. Measured residual against the requested pose: 1.6 mm, 0.09 deg, 0.03 deg roll.
        """
        import mujoco
        _planner, env, _indices = self.planner()
        transform = np.asarray(world_from_camera, float)
        position, up, forward = transform[:3, 3], -transform[:3, 1], transform[:3, 2]

        model = env.model
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(width))
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(height))
        model.vis.global_.ipd = 0.0
        model.vis.global_.fovy = float(fovy_deg)

        key = (int(width), int(height))
        if self._renderer is not None and self._renderer[0] != key:
            self._renderer[1].close()
            self._renderer = None
        if self._renderer is None:
            self._renderer = (key, mujoco.Renderer(model, height=int(height), width=int(width)))
        renderer = self._renderer[1]

        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 1.0
        camera.lookat[:] = position + forward * camera.distance
        camera.elevation = float(np.degrees(np.arcsin(np.clip(forward[2], -1.0, 1.0))))
        camera.azimuth = float(np.degrees(np.arctan2(up[1], up[0])))
        renderer.update_scene(env.data, camera=camera)
        return renderer.render()

    def fling(self, fling_vector, distance_fwd, distance_snap, label='fling'):
        """Execute the planner's native fling: forward sweep, then a snap back that ends just above the table.

        Industrial18DOFPlanner.plan_fling_motion already works in cage world (its end height is TABLE_Z +
        FLING_END_HEIGHT_ABOVE_TABLE) and starts from the current TCPs, so the fling needs no frame conversion --
        unlike real_world/fling.py, whose literals are per-UR5-base-frame for a floor-standing arm.
        """
        planner, _env, _indices = self.planner()
        traj = planner.plan_fling_motion(np.asarray(fling_vector, float), distance_fwd, distance_snap)
        if traj is None:
            raise MotionPlanningError(f'{label}: no collision-free / torque-feasible fling from the current pose')
        return self.execute(traj, label=label)

    def plan_arm_joints(self, q_l, q_r, label='arm move'):
        """Arms-only joint move with active collision avoidance, carriages left where they are.

        plan_simple_ptp interpolates straight through joint space and only *checks* the result, so a large arm swing
        between two valid configurations fails whenever the straight line clips the table. plan_joint_to_joint pushes
        the path off obstacles with a collision gradient instead, which is what a big reconfiguration needs.
        """
        planner, _env, _indices = self.planner()
        traj = planner.plan_joint_to_joint(np.asarray(q_l, float), np.asarray(q_r, float))
        if traj is None:
            raise MotionPlanningError(f'{label}: no collision-free arm path from the current state')
        return traj

    def hover_configuration(self):
        """Joint configuration for hover-ready, derived by IK from the hover TCP targets. Cached.

        Solved with the model seeded at the OBSERVATION pose, not from wherever the arms currently are. IK returns
        whichever branch is nearest its seed, and a hover solved from the hanging pose came back with Bernoulli's
        joint 1 at +1.384 rad against the observation pose's -0.567: interpolating between those two rotates the
        arm the long way round and drives the gripper 86 mm into cage_col_side_left at the halfway point. Seeding
        from the observation pose keeps hover in the same branch, so observe -> hover -> observe stays a short move.

        The target itself is Cartesian on purpose (flingbot_config.hover_tcp_targets) so it tracks TABLE_Z; only
        the branch is pinned.
        """
        if self._hover_config is not None:
            return self._hover_config
        import mujoco
        from gemsort_lib.flingbot_config import hover_tcp_targets, QUAT_DOWN
        _planner, env, indices = self.planner()
        _load_camera_pose()
        if CAMERA_POSE_ARM is None or CAMERA_POSE_RAIL is None:
            raise RuntimeError('cannot solve the hover configuration: no taught observation pose to seed IK from')
        want_l, want_r = hover_tcp_targets()
        params = [list(want_l) + list(QUAT_DOWN), list(want_r) + list(QUAT_DOWN)]
        # Seed BOTH chains explicitly. IK converges to whichever branch is nearest its seed, so inheriting
        # whatever the right arm happened to be doing would make the hover solution -- and therefore which
        # transitions are reachable from it -- depend on when it was first requested.
        from gemsort_lib.flingbot_config import (HANGER_POSE_R, LIN_AXIS_X_R, LIN_AXIS_Y_CENTER)
        backup = env.data.qpos.copy()
        try:
            env.data.qpos[:] = 0
            env.data.qpos[indices['left_arm']] = CAMERA_POSE_ARM
            env.data.qpos[indices['laxis_left']] = CAMERA_POSE_RAIL
            env.data.qpos[indices['right_arm']] = HANGER_POSE_R
            env.data.qpos[indices['laxis_right']] = [LIN_AXIS_X_R, LIN_AXIS_Y_CENTER]
            mujoco.mj_forward(env.model, env.data)
            traj = self.plan(params, mode='joint', label='solve hover configuration')
        finally:
            env.data.qpos[:] = backup
            mujoco.mj_forward(env.model, env.data)
        self._hover_config = {'q_l': np.asarray(traj['q_l'][-1], float), 'q_r': np.asarray(traj['q_r'][-1], float),
                              'lin_l': np.asarray(traj['lin_l'][-1], float),
                              'lin_r': np.asarray(traj['lin_r'][-1], float)}
        logger.info('hover configuration solved (seeded from the observation pose): q_l %s, rail %s',
                    np.round(self._hover_config['q_l'], 3), np.round(self._hover_config['lin_l'], 3))
        return self._hover_config

    def go_to_hover(self, label='hover-ready'):
        """Move both arms to the hover-ready configuration, carriages first when a direct path is blocked."""
        if self.where_are_the_arms()[0] == 'hover':
            return True
        hover = self.hover_configuration()
        try:
            traj = self.plan_ptp(hover['q_l'], hover['q_r'], hover['lin_l'], hover['lin_r'], label=label)
            return self.execute(traj, label=label)
        except MotionPlanningError as error:
            # Staged fallback, the same shape move_flingbot.prepare_workspace() uses: drive the carriages with the
            # arms still folded, then unfold them in place. Escaping the hanging rest pose needs this -- a direct
            # interpolation from hanging to hover has no collision-free straight line through joint space.
            logger.info('direct path to %s blocked (%s); staging carriages then arms.', label, error)
            state = self.joint_state()
            traj = self.plan_ptp(state['q_l'], state['q_r'], hover['lin_l'], hover['lin_r'],
                                 label=f'{label} (1/2: carriages)')
            if not self.execute(traj, label=f'{label} (1/2: carriages)'):
                return False
            traj = self.plan_arm_joints(hover['q_l'], hover['q_r'], label=f'{label} (2/2: unfold arms)')
            return self.execute(traj, label=f'{label} (2/2: unfold arms)')

    def go_to_observation(self, label='observation pose'):
        """Put Bernoulli where the camera was taught and Pascal at its mirrored park, as one planned 18-DOF move."""
        _load_camera_pose()
        if CAMERA_POSE_ARM is None or CAMERA_POSE_RAIL is None:
            raise RuntimeError('no taught Bernoulli observation pose to move to')
        if self.where_are_the_arms()[0] == 'observation':
            return True
        # Only Bernoulli moves. Pascal has no camera and nothing to contribute to an observation, and its hover
        # station ([0.7, 0.58, 1.04]) already sits well outside the wrist camera's cone -- moving it toward the
        # table centre to "park" it would put the gripper straight under the lens and occlude the cloth.
        #
        # Staged carriage-then-arm, like move_flingbot.prepare_workspace(): plan_simple_ptp interpolates straight
        # through joint space with no avoidance, so asking it to swing the arm and drive the carriage at once from
        # hover to the observation pose puts the gripper through the table on the way.
        state = self.joint_state()
        traj = self.plan_ptp(state['q_l'], state['q_r'], CAMERA_POSE_RAIL, state['lin_r'],
                             label=f'{label} (1/2: carriage)')
        if not self.execute(traj, label=f'{label} (1/2: carriage)'):
            return False
        state = self.joint_state()
        traj = self.plan_arm_joints(CAMERA_POSE_ARM, state['q_r'], label=f'{label} (2/2: arm)')
        return self.execute(traj, label=f'{label} (2/2: arm)')

    def return_to_hover(self, label='back to hover-ready'):
        """Bring Bernoulli back off the observation pose so the pair is ready to grasp."""
        return self.go_to_hover(label=label)

    def prepare_cell(self):
        """Bring the cell to hover-ready before the policy starts, from wherever the arms happen to be.

        Called once when the operator starts a run. Reports what it found, and only moves when it has to, so
        starting a run that is already staged costs nothing.
        """
        where, state = self.where_are_the_arms()
        logger.info('cell state at start: %s (left rail %s, right rail %s)',
                    where.upper(), np.round(state['lin_l'], 3), np.round(state['lin_r'], 3))
        if where == 'hover':
            logger.info('already hover-ready; nothing to stage')
            return True
        if where == 'hanging':
            logger.info('arms are hanging: staging them over the table before the run')
        return self.go_to_hover(label=f'stage to hover-ready (from {where})')

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
            if self.grippers is not None:
                self.grippers.env = env
                self.grippers._planner_handles.clear()
                for side, width in self.grippers._current_widths.items():
                    self.grippers._set_planner_width(side, width)
            self._planner = (Industrial18DOFPlanner(env, indices), env, indices)
            self._seed_planning_state(env, indices)
        return self._planner

    def _seed_planning_state(self, env, indices):
        """Put the planning model into a real, reachable configuration before anything is planned from it.

        A freshly compiled MuJoCo model sits at qpos = 0, which for this cell is not a neutral pose: both rail
        carriages land at world x ~= 0, outside their own travel (left 1.660..2.580, right 0.410..1.260), stacked on
        each other inside the cage frame -- 98 penetrating contacts, with the arm 121 mm inside the cage wall. Every
        trajectory planned from there begins in collision, so _validate_final_trajectory rejects it within the first
        few steps whatever the goal was, and the failure reads as "unreachable" rather than "bad start state".

        The single-arm pipeline (move_flingbot.py) calls sync_planning_env_to_robot before each plan. This path only
        called it AFTER executing a trajectory, so the first plan never had a valid start.
        """
        import mujoco

        if not self._dry_run() and not isinstance(self.bot, _DryRunController):
            from gemsort_lib.flingbot_controller import sync_planning_env_to_robot
            sync_planning_env_to_robot(env, self.bot, indices)
            return

        # Dry run opens no sockets, so there is no live state to read. Assume the cell's REST state: both arms
        # hanging off their gantries at the rail start positions. That is how the cell powers up and how the
        # operator leaves it, and it is the state prepare_cell() has to stage out of -- so dry runs exercise the
        # same path a real run takes instead of a fiction that is already staged.
        from gemsort_lib.flingbot_config import (HANGER_POSE_L, HANGER_POSE_R, LIN_AXIS_X_L, LIN_AXIS_X_R,
                                                 LIN_AXIS_Y_CENTER)
        env.data.qpos[:] = 0
        env.data.qpos[indices['left_arm']] = HANGER_POSE_L
        env.data.qpos[indices['right_arm']] = HANGER_POSE_R
        env.data.qpos[indices['laxis_left']] = [LIN_AXIS_X_L, LIN_AXIS_Y_CENTER]
        env.data.qpos[indices['laxis_right']] = [LIN_AXIS_X_R, LIN_AXIS_Y_CENTER]
        mujoco.mj_forward(env.model, env.data)
        logger.info('planning model seeded with the cell at rest (both arms hanging at the rail start positions)')
        _warn_if_start_state_unusable(env)


    def move(self, move_type, params, blocking=True, **kwargs):
        return getattr(self, f'move{move_type}')(params, blocking=blocking, **kwargs)

    def check_action_reachability(self, left_pose, right_pose, **kwargs):
        """FlingBot asks this before executing a grasp pair; plans both arms, moves nothing."""
        return self.reachable([left_pose, right_pose], **kwargs)

    def close_grippers(self, blocking=True, **kwargs):
        if self._dry_run():
            logger.warning('close_grippers skipped (GEMSORT_DRY_RUN=1)')
            return False
        self.bot.grasp_grippers(side='both', wait=blocking)
        return True

    def open_grippers(self, blocking=True, **kwargs):
        if self._dry_run():
            logger.warning('open_grippers skipped (GEMSORT_DRY_RUN=1)')
            return False
        self.bot.release_grippers(side='both', wait=blocking)
        return True

    def out_of_the_way(self):
        # Never translate an upstream UR5 "out of the way" request into an
        # unplanned dual-arm joint move. GEMSORT recovery/observation motion
        # must go through the 18-DOF collision planner.
        logger.warning('out_of_the_way skipped: no validated 18-DOF observation trajectory is configured')

    # ---------------------------------------------------------------- gemsort extras
    def go_to_camera_pose(self):
        """Park Bernoulli so the wrist RealSense looks straight down at the table centre, Pascal out of view."""
        _load_camera_pose()
        if CAMERA_POSE_ARM is None or CAMERA_POSE_RAIL is None:
            raise RuntimeError('no observation pose taught yet — jog Bernoulli over the table and run '
                               '`python -m real_world.gemsort_calibrate --save-camera-pose` (or the GUI button)')
        if self._dry_run():
            logger.warning('camera-pose motion skipped (GEMSORT_DRY_RUN=1)')
            return False
        raise RuntimeError('automatic camera-pose motion is disabled until a collision-validated 18-DOF '
                           'observation trajectory is implemented; use Manual Robot Control')

    def camera_world_transform(self):
        """World-from-D435 transform using live kinematics + hand-eye calibration."""
        import json
        import pathlib
        import mujoco
        planner, env, indices = self.planner()
        _load_camera_pose()
        if CAMERA_POSE_ARM is None or CAMERA_POSE_RAIL is None:
            raise RuntimeError('no saved Bernoulli observation pose')
        # Evaluate the camera pose in the kinematic model only. Do not move or
        # read either robot; dry-run must be completely command-free.
        # Bernoulli carries the RealSense and is the LEFT side (see the module
        # docstring); Pascal/right only has to be parked out of view.
        _check_within_joint_range(env.model, indices['left_arm'], CAMERA_POSE_ARM, 'Bernoulli observation arm')
        _check_within_joint_range(env.model, indices['laxis_left'], CAMERA_POSE_RAIL, 'Bernoulli observation rail')
        # This is a pure query: restore whatever start state the planner was seeded with, or a plan issued after
        # this call would silently begin from the observation pose instead of the cell's actual configuration.
        backup = env.data.qpos.copy()
        try:
            # Only the left chain matters: the transform is read off Bernoulli's flange. Pascal's configuration
            # cannot affect it, so do not seed it from park data that may be stale.
            env.data.qpos[indices['left_arm']] = CAMERA_POSE_ARM
            env.data.qpos[indices['laxis_left']] = CAMERA_POSE_RAIL
            mujoco.mj_forward(env.model, env.data)
            body_name = 'cage/iiwa/iiwa_link_7'  # Bernoulli/left physical flange
            body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise RuntimeError(f'planner model has no Bernoulli flange body {body_name!r}')
            world_from_flange = np.eye(4)
            world_from_flange[:3, :3] = env.data.xmat[body_id].reshape(3, 3)
            world_from_flange[:3, 3] = env.data.xpos[body_id]
        finally:
            env.data.qpos[:] = backup
            mujoco.mj_forward(env.model, env.data)

        root = pathlib.Path(__file__).resolve().parents[2]
        calibration_file = root.parent / 'gemsort_demo_tool/data_capture/calibration/wrist_camera_extrinsics.json'
        calibration = json.loads(calibration_file.read_text())
        camera = calibration['cameras']['realsense_d435']
        flange_from_camera = np.asarray(camera['matrix_flange_from_camera'], float)
        transform = world_from_flange @ flange_from_camera
        logger.info('D435 world transform from %s (serial %s)', calibration_file, camera.get('serial'))
        _warn_if_not_overhead(transform)
        return transform


def side_config(side, sim):
    """Same connection config move_flingbot.get_side_config builds (ports/IPs from gemsort_lib.flingbot_config)."""
    from gemsort_lib import flingbot_config as cfg
    if sim:
        ports = (cfg.SIM_PORT_IIWA_L, cfg.SIM_PORT_RAIL_L) if side == 'left' else (cfg.SIM_PORT_IIWA_R, cfg.SIM_PORT_RAIL_R)
        local = robot_ip = cfg.LOCAL_IP
    else:
        ports = (cfg.PORT_IIWA_L, cfg.PORT_RAIL_L) if side == 'left' else (cfg.PORT_IIWA_R, cfg.PORT_RAIL_R)
        local = cfg.REAL_LOCAL_IP_LEFT if side == 'left' else cfg.REAL_LOCAL_IP_RIGHT
        robot_ip = cfg.REAL_ROBOT_IP_LEFT if side == 'left' else cfg.REAL_ROBOT_IP_RIGHT
    iiwa, laxis = ports
    return {'local_ip_addr': local, 'robot_ip_addr': robot_ip,
            'iiwa': {'robot_port': iiwa[0], 'client_port': iiwa[1]},
            'laxis': {'robot_port': laxis[0], 'client_port': laxis[1]}}


def _mode_is_simulated(name, legacy_default=False):
    value = os.environ.get(name)
    if value is None:
        return legacy_default
    value = value.strip().lower()
    if value not in {'real', 'simulated', 'sim'}:
        raise ValueError(f'{name} must be "real" or "simulated", got {value!r}')
    return value != 'real'


def simulation_modes(sim=None, simulation_left=None, simulation_right=None):
    """Resolve Pascal/left and Bernoulli/right modes, preserving GEMSORT_SIM compatibility."""
    legacy = (os.environ.get('GEMSORT_SIM', '0') == '1') if sim is None else bool(sim)
    left = (_mode_is_simulated('GEMSORT_PASCAL_MODE', legacy)
            if simulation_left is None else bool(simulation_left))
    right = (_mode_is_simulated('GEMSORT_BERNOULLI_MODE', legacy)
             if simulation_right is None else bool(simulation_right))
    return left, right


class _OfflineSide:
    """Identity placeholder used when dry-run intentionally opens no ARC sockets."""


class _DryRunController:
    """Fail closed if a dry-run path accidentally attempts controller I/O."""
    left = _OfflineSide()
    right = _OfflineSide()

    def __getattr__(self, name):
        raise RuntimeError(f'dry-run attempted forbidden robot-controller operation {name!r}')


def connect(sim=None, real_grippers=True, planner_env=None,
            simulation_left=None, simulation_right=None):
    """(FlingBotController, DualGripperManager) for the GEMSORT cell; import-time heavy, hence lazy."""
    from arcpy.robots import Robot18DOF
    from gemsort_lib.flingbot_controller import FlingBotController
    from gemsort_lib.flingbot_gripper import DualGripperManager
    simulation_left, simulation_right = simulation_modes(
        sim=sim, simulation_left=simulation_left, simulation_right=simulation_right)
    logger.info('18-DOF modes: Pascal/left=%s, Bernoulli/right=%s',
                'simulated' if simulation_left else 'real',
                'simulated' if simulation_right else 'real')
    if os.environ.get('GEMSORT_DRY_RUN', '1') == '1':
        logger.warning('dry-run: not opening ARC robot sockets or gripper connections')
        return _DryRunController(), None
    robot = Robot18DOF({
        'left': side_config('left', simulation_left),
        'right': side_config('right', simulation_right),
    })
    controller = FlingBotController(robot)
    grippers = DualGripperManager(
        planner_env,
        simulation_left=simulation_left,
        simulation_right=simulation_right,
        use_real_gripper_left=real_grippers and not simulation_left,
        use_real_gripper_right=real_grippers and not simulation_right,
    )
    controller.attach_grippers(grippers)
    return controller, grippers


def main():
    p = argparse.ArgumentParser(description='GEMSORT arm pair helper for the FlingBot real-world stack')
    p.add_argument('--camera-pose', action='store_true', help='move Bernoulli into the overhead RealSense pose')
    p.add_argument('--home', action='store_true', help='move both arms to the home pose')
    p.add_argument('--save-camera-pose', action='store_true', help='store the CURRENT pose as the observation pose')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    pair = GemsortArmPair()
    if args.home:
        pair.homej()
    if args.save_camera_pose:
        from real_world.gemsort_calibrate import save_camera_pose
        save_camera_pose(pair)
    if args.camera_pose:
        pair.go_to_camera_pose()
    logger.info('done')


if __name__ == '__main__':
    main()
