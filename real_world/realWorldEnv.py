import sys
import types

# RealWorldEnv reuses SimEnv's observation/action-selection methods, but the
# upstream SimEnv module imports PyFlex at module load time. No PyFlex API is
# used on this path because setup_env and all motion primitives are overridden.
try:
    import pyflex  # noqa: F401
except ImportError:
    sys.modules['pyflex'] = types.ModuleType('pyflex')

from real_world import (
    UR5Pair, UR5MoveTimeoutException,
    fling, stretch, pick_and_drop)
from real_world.setup import (
    get_top_cam, get_front_cam,
    CLOTHS_DATASET, CURRENT_CLOTH,
    DEFAULT_ORN, DIST_UR5, WS_PC,
    MIN_GRASP_WIDTH, MAX_GRASP_WIDTH,
    WORKSPACE_SURFACE, MAX_GRASP_HEIGHT_ABOVE_SURFACE,)
from real_world.stretch import is_cloth_grasped
from real_world.utils import (

    pix_to_3d_position, get_workspace_crop,
    get_cloth_mask, compute_coverage,
    bound_grasp_pos)
from environment.simEnv import SimEnv
from environment.Memory import Memory
from environment.tasks import Task

from learning.nets import prepare_image
from time import sleep, strftime
from real_world.realur5_utils import setup_thread
from environment.utils import (
    preprocess_obs,
    add_text_to_image)
from filelock import FileLock
from copy import deepcopy
import numpy as np
from time import time
import os
import h5py
import cv2


class GraspFailException(Exception):
    def __init__(self):
        super().__init__('Grasp failed due to real world')


class RealWorldEnv(SimEnv):
    def __init__(self, replace_background=True, **kwargs):
        global CURRENT_CLOTH
        self.replace_background = replace_background
        cloth_name = os.environ.get('GEMSORT_CLOTH_NAME', CURRENT_CLOTH).strip() or CURRENT_CLOTH
        cloth_width = float(os.environ.get('GEMSORT_CLOTH_WIDTH_M', '.45'))
        cloth_length = float(os.environ.get('GEMSORT_CLOTH_LENGTH_M', '.55'))
        cloth_mass = float(os.environ.get('GEMSORT_CLOTH_MASS_KG', '.20'))
        if cloth_width <= 0 or cloth_length <= 0 or cloth_mass <= 0:
            raise ValueError('cloth width, length and mass must all be positive')
        template = dict(CLOTHS_DATASET.get(CURRENT_CLOTH, next(iter(CLOTHS_DATASET.values()))))
        template.update(cloth_size=(cloth_width, cloth_length), mass=cloth_mass)
        CLOTHS_DATASET[cloth_name] = template
        CURRENT_CLOTH = cloth_name
        print(f'[gemsort] current cloth: {CURRENT_CLOTH!r}, {cloth_width:.3f} x {cloth_length:.3f} m, '
              f'{cloth_mass:.3f} kg')

        def randomize_cloth():
            # The upstream reset routine directly commands right_ur5/left_ur5
            # without GEMSORT's 18-DOF collision planner. The operator places
            # the garment for GEMSORT evaluation; reset must not move robots.
            if os.environ.get('GEMSORT_ROBOTS', '1') != '1':
                pick_and_drop(
                    ur5_pair=self.ur5_pair, top_camera=self.top_cam,
                    top_cam_right_ur5_pose=self.top_cam_right_ur5_pose,
                    top_cam_left_ur5_pose=self.top_cam_left_ur5_pose,
                    cam_depth_scale=self.cam_depth_scale)
                self.ur5_pair.out_of_the_way()
            return Task(
                name=f'{CURRENT_CLOTH}' +
                strftime("%Y-%m-%d_%H-%M-%S"),
                flatten_area=CLOTHS_DATASET[CURRENT_CLOTH]['flatten_area'],
                initial_coverage=self.compute_coverage(),
                task_difficulty='hard',
                cloth_mass=CLOTHS_DATASET[CURRENT_CLOTH]['mass'],
                cloth_size=CLOTHS_DATASET[CURRENT_CLOTH]['cloth_size'],
            )
        super().__init__(
            get_task_fn=randomize_cloth,
            parallelize_prepare_image=True,
            episode_length=10,
            **kwargs)
        # state variables to handle recorder
        self.recording = False
        self.recording_daemon = None
        np.random.seed(int(time()))
        self.action_handlers = {
            'fling': self.pick_and_fling_primitive,
            'drag': self.pick_and_drag_primitive,
            'place': self.pick_and_place_primitive
        }

    def setup_env(self):
        # used for getting obs
        self.top_cam = get_top_cam()

        # used for stretching primitive
        self.front_cam = get_front_cam()
        if self.front_cam is None:
            print('[gemsort] no front camera configured: cloth-contact verification and vision-guided stretching '
                  'will be skipped')

        # used for recording visualizations
        # if you have a third camera/webcam,
        # you can setup a different camera here
        # with a better view of both arms
        self.setup_cam = None

        # GEMSORT: two rail-mounted iiwa (2 x 9 DoF) instead of two UR5s. Set GEMSORT_ROBOTS=0 to fall back to the
        # original UR5 backend; everything below keeps calling the same UR5Pair API.
        import os
        if os.environ.get('GEMSORT_ROBOTS', '1') == '1':
            from real_world.gemsort_arm_pair import GemsortArmPair
            self.ur5_pair = GemsortArmPair()
        else:
            self.ur5_pair = UR5Pair()
        self.ur5_pair.open_grippers()
        # The UR5 implementation loaded fixed text files here. Bernoulli's
        # camera is wrist-mounted, so derive its world pose from current robot
        # kinematics and the measured D435 hand-eye transform.
        # Stage the cell before anything else looks at it. The arms normally sit in the hanging rest pose, from
        # which no Cartesian move across the table is plannable -- so the run has to bring them over the table
        # first, or every grasp comes back as "unreachable" for reasons that have nothing to do with the grasp.
        if os.environ.get('GEMSORT_ROBOTS', '1') == '1':
            self.ur5_pair.prepare_cell()

        camera_world = self.ur5_pair.camera_world_transform()
        self.top_cam_right_ur5_pose = camera_world
        self.top_cam_left_ur5_pose = camera_world
        self.cam_depth_scale = 1.0  # GemsortRealSense already returns metres.

    def _ensure_observing(self):
        """Put Bernoulli back on the taught observation pose before any frame is grabbed.

        camera_world_transform() builds the pixel-to-world extrinsic from the STORED observation pose, and the
        workspace crop is calibrated for that same viewpoint, so a frame taken from anywhere else reconstructs to
        the wrong place. Cheap when already there: go_to_observation() short-circuits without planning.
        """
        if os.environ.get('GEMSORT_ROBOTS', '1') == '1':
            self.ur5_pair.go_to_observation()

    def get_cloth_mask(self, rgb=None, depth=None):
        if rgb is None:
            self._ensure_observing()
            rgb, depth = self.top_cam.get_rgbd()
        return get_cloth_mask(rgb, depth=depth)

    def preaction(self):
        self.preaction_mask = self.get_cloth_mask()

    def compute_iou(self):
        mask = self.get_cloth_mask()
        intersection = np.logical_and(
            mask, self.preaction_mask).sum()
        union = np.logical_or(mask, self.preaction_mask).sum()
        return intersection/union

    def postaction(self):
        iou = self.compute_iou()
        print(f'\tIoU: {iou:.04f}')
        if iou > 1 - 1e-1:
            self.terminate = True

    def get_max_value_valid_action(self, value_maps):
        """GEMSORT: same choice as FlingBot, but draw it on the live camera image first and let the operator gate it.

        Shows the predicted grasp pair (pretransform pixels, i.e. the frame self.pretransform_rgb is in) over the
        value map. With GEMSORT_CONFIRM=1 an aborted action returns (None, None), which FlingBot treats as "no action".
        """
        from real_world import gemsort_viz
        action_primitive, action = super().get_max_value_valid_action(value_maps)
        if action is None:
            gemsort_viz.show(getattr(self, 'pretransform_rgb', None), status='no valid action found')
            return action_primitive, action
        pixels = np.array(action.get('pretransform_pixels', []), dtype=float).reshape(-1, 2)[:, ::-1]  # (row, col) → (x, y)
        status = f"step {self.current_timestep}: {action_primitive}, scale {action.get('scale', float('nan')):.2f}"
        if not gemsort_viz.confirm(getattr(self, 'pretransform_rgb', None), status=status,
                                   grasp_pixels=pixels if len(pixels) == 2 else None,
                                   value_map=np.asarray(action.get('value_map')) if action.get('value_map') is not None else None):
            print('\t[GEMSORT] action aborted by operator')
            return None, None
        # the backend shows this frame in its confirmation window (plain attribute: harmless for the UR5 fallback)
        self.ur5_pair.preview = getattr(self, 'pretransform_rgb', None)
        return action_primitive, action

    def step(self, value_maps):
        # NOTE: negative current_timestep's
        # act as error codes
        print(f'Step {self.current_timestep}')
        try:
            retval = super().step(value_maps)
            self.episode_memory.add_value(
                key='failed_grasp', value=0)
            self.episode_memory.add_value(
                key='timed_out', value=0)
            self.episode_memory.add_value(
                key='cloth_stuck', value=0)
            return retval
        except GraspFailException as e:
            # action failed in real world
            print('\t[ERROR]', e)
            current_timestep = self.current_timestep
            self.current_timestep = -2
            self.ur5_pair.open_grippers()
            self.ur5_pair.out_of_the_way()
            if self.dump_visualizations:
                self.stop_recording()
            self.current_timestep = current_timestep
            # remove previous observation
            del self.episode_memory.data['observations'][-1]
            self.episode_memory.data['failed_grasp'] = [
                1] * len(self.episode_memory)
            print(f'\tNow episode has {len(self.episode_memory)} steps')
            self.on_episode_end()
            return self.reset()
        except UR5MoveTimeoutException as e:
            # action failed in real world
            print('\t[ERROR]', e)
            current_timestep = self.current_timestep
            self.current_timestep = -4
            self.ur5_pair.open_grippers()
            self.ur5_pair.out_of_the_way()
            if self.dump_visualizations:
                self.stop_recording()
            self.current_timestep = current_timestep
            # remove previous observation
            del self.episode_memory.data['observations'][-1]
            self.episode_memory.data['timed_out'] = [
                1] * len(self.episode_memory)
            print(f'\tNow episode has {len(self.episode_memory)} steps')
            self.on_episode_end()
            return self.reset()

    def start_recording(self):
        if self.recording:
            return
        self.recording = True
        self.recording_daemon = setup_thread(
            target=self.record_video_daemon_fn)

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self.recording_daemon.join()
        self.recording_daemon = None

    def record_video_daemon_fn(self):
        while self.recording:
            # NOTE: negative current_timestep's
            # act as error codes
            text = f'step {self.current_timestep}'
            if self.current_timestep == -1:
                text = 'randomizing cloth'
            elif self.current_timestep == -2:
                text = 'grasp failed'
            elif self.current_timestep == -3:
                text = 'cloth stuck'
            elif self.current_timestep == -4:
                text = 'ur5 timed out'
            if 'top' not in self.env_video_frames:
                self.env_video_frames['top'] = []
            top_view = cv2.resize(
                get_workspace_crop(
                    self.top_cam.get_rgbd()[0].copy()),
                (256, 256))
            self.env_video_frames['top'].append(
                add_text_to_image(
                    image=top_view,
                    text=text, fontsize=16))
            if self.setup_cam is not None:
                if 'setup' not in self.env_video_frames:
                    self.env_video_frames['setup'] = []
                self.env_video_frames['setup'].append(
                    self.setup_cam.get_rgbd(repeats=1)[0])
            if len(self.env_video_frames['setup']) > 50000:
                # episode typically ends in 4000 frames
                print('Robot probably got into error... Terminating')
                exit()

    def pick_and_fling_primitive(self, p1, p2, grasp_width,
                                 p1_grasp_cloth: bool, p2_grasp_cloth: bool, fling_height=0.25):
        if os.environ.get('GEMSORT_ROBOTS', '1') != '1':
            return self._pick_and_fling_upstream(p1, p2, grasp_width, p1_grasp_cloth, p2_grasp_cloth, fling_height)
        return self._pick_and_fling_gemsort(p1, p2)

    def _pick_and_fling_gemsort(self, left_point, right_point):
        """Cage-world pick-and-fling, following move_flingbot.perform_garment_manipulation.

        real_world/fling.py is NOT used here. Its waypoints are expressed in each UR5's own base frame, with x
        reaching toward the opposite robot and z measured up from a floor-standing base; none of that transforms
        onto a pair of ceiling-mounted iiwas. The GEMSORT planner already implements the fling natively in cage
        world, so this drives that instead of rewriting the literals.
        """
        from gemsort_lib.flingbot_config import (PRE_GRASP_HEIGHT, GRIPPER_OFFSET, LIFT_HEIGHT, TILT_ANGLE,
                                                 STRETCH_FACTOR, TABLE_CENTER_X, TABLE_CENTER_Y,
                                                 FLING_FWD, FLING_BACK)
        from gemsort_lib.flingbot_utils import calculate_tilted_quats, scipy_to_mujoco_quat
        from scipy.spatial.transform import Rotation

        # Off the observation pose first: the grasp is planned from wherever the arms are, and observing leaves
        # Bernoulli folded over the table centre.
        if not self.ur5_pair.return_to_hover():
            return

        left = np.asarray(left_point, float)
        right = np.asarray(right_point, float)
        q_l, q_r = calculate_tilted_quats(left, right, tilt_angle_deg=-TILT_ANGLE, ee_rot_deg=90.0)

        def pair(left_xyz, right_xyz, quat_l, quat_r):
            return [list(left_xyz) + list(quat_l), list(right_xyz) + list(quat_r)]

        up = np.array([0.0, 0.0, 1.0])
        steps = (
            ('pre-grasp', pair(left + up * PRE_GRASP_HEIGHT, right + up * PRE_GRASP_HEIGHT, q_l, q_r), 'joint'),
            ('descent',   pair(left + up * GRIPPER_OFFSET,   right + up * GRIPPER_OFFSET,   q_l, q_r), 'cartesian'),
        )
        for label, params, mode in steps:
            if not self.ur5_pair.movel(params, mode=mode, label=label):
                return
        self.ur5_pair.close_grippers()
        self._show_grasp_verification(left, right)

        if not self.ur5_pair.movel(pair(left + up * LIFT_HEIGHT, right + up * LIFT_HEIGHT, q_l, q_r),
                                   label='lift'):
            return

        # Stretch and centre the garment over the table before flinging, so the fling always starts from the same
        # place regardless of where on the table the grasp happened to be.
        stretched = float(np.linalg.norm(left - right)) * (1.0 + STRETCH_FACTOR)
        z_target = float(left[2]) + LIFT_HEIGHT
        align_l = np.array([TABLE_CENTER_X + stretched / 2.0, TABLE_CENTER_Y, z_target])
        align_r = np.array([TABLE_CENTER_X - stretched / 2.0, TABLE_CENTER_Y, z_target])
        delta = align_r - align_l
        yaw = np.degrees(np.arctan2(delta[1], delta[0]))
        q_align = scipy_to_mujoco_quat((Rotation.from_euler('z', yaw, degrees=True)
                                        * Rotation.from_euler('y', 180.0, degrees=True)
                                        * Rotation.from_euler('z', 90.0, degrees=True)).as_quat())
        if not self.ur5_pair.movel(pair(align_l, align_r, q_align, q_align), label='stretch & align'):
            return

        # Fling along +y: the arms are separated along world x, so y is the free sweep direction.
        self.ur5_pair.fling(np.array([0.0, 1.0, 0.0]), FLING_FWD, FLING_BACK, label='fling')
        self.ur5_pair.open_grippers()
        self.ur5_pair.return_to_hover(label='post-fling: back to hover')

    def _pick_and_fling_upstream(
            self, p1, p2,
            grasp_width,
            p1_grasp_cloth: bool,
            p2_grasp_cloth: bool,
            fling_height=0.25):
        """Upstream UR5 implementation: per-UR5-base-frame waypoints, used only when GEMSORT_ROBOTS=0."""
        left_point, right_point = p1, p2
        left_point = bound_grasp_pos(left_point)
        right_point = bound_grasp_pos(right_point)

        if self.dump_visualizations:
            self.start_recording()
        # grasp cloth
        self.ur5_pair.movel(
            params=[
                # left
                left_point + DEFAULT_ORN,
                # right
                right_point + DEFAULT_ORN],
            blocking=True, use_pos=True)
        self.ur5_pair.close_grippers()
        # this slow backup help with grasp
        # success in pinch grasping cloths
        left_point[-1] += 0.03
        right_point[-1] += 0.03
        self.ur5_pair.movel(
            params=[
                # left
                left_point + DEFAULT_ORN,
                # right
                right_point + DEFAULT_ORN],
            blocking=True, use_pos=True,
            j_vel=0.01, j_acc=0.01)
        self.ur5_pair.close_grippers()
        # lift cloth
        dx = (DIST_UR5 - grasp_width)/2
        self.ur5_pair.movel(
            params=[
                # left
                [dx, 0,  fling_height] + DEFAULT_ORN,
                # right
                [dx, 0,  fling_height] + DEFAULT_ORN],
            blocking=True, use_pos=True)
        if self.front_cam is None:
            # Preserve FlingBot's pixel-mask expectation for which side should
            # contain cloth, but do not claim that a camera verified it.
            left_grasping, right_grasping = bool(p2_grasp_cloth), bool(p1_grasp_cloth)
        else:
            left_grasping, right_grasping = is_cloth_grasped(
                depth=self.front_cam.get_rgbd()[1])
        if (p1_grasp_cloth and not right_grasping)\
                or (p2_grasp_cloth and not left_grasping):
            raise GraspFailException
        if (left_grasping or right_grasping):
            if left_grasping and right_grasping and self.front_cam is not None:
                grasp_width = stretch(
                    ur5_pair=self.ur5_pair,
                    front_camera=self.front_cam,
                    height=fling_height, grasp_width=grasp_width)
            if self.front_cam is not None:
                left_grasping, right_grasping = is_cloth_grasped(
                    depth=self.front_cam.get_rgbd()[1])
            fling(ur5_pair=self.ur5_pair,
                  height=fling_height,
                  grasp_width=grasp_width,
                  left_grasping=left_grasping,
                  right_grasping=right_grasping)
        else:
            self.terminate = True
        self.ur5_pair.open_grippers()
        self.ur5_pair.out_of_the_way()
        if self.dump_visualizations:
            self.stop_recording()

    def pick_and_drag_primitive(self, **kwargs):
        raise NotImplementedError()

    def pick_stretch_drag_primitive(self, **kwargs):
        raise NotImplementedError()

    def pick_and_place_primitive(self, p1, p2, info, height=0.2, **kwargs):
        if os.environ.get('GEMSORT_ROBOTS', '1') == '1':
            raise NotImplementedError('pick/place is disabled for GEMSORT: the upstream implementation directly '
                                      'commands individual UR5s; only the collision-planned fling primitive is enabled')
        pick_point, place_point = p1, p2
        pick_point = bound_grasp_pos(pick_point)
        place_point = bound_grasp_pos(place_point)

        arm = info['which_arm']
        should_grasp_cloth = info['should_grasp_cloth']

        if self.dump_visualizations:
            self.start_recording()
        prepick_point = deepcopy(pick_point)
        backup_point = deepcopy(pick_point)
        prepick_point[2] += 0.05
        backup_point[2] += 0.02
        preplace_point = deepcopy(place_point)
        preplace_point[2] += 0.05
        if arm == 'left':
            self.ur5_pair.left_ur5.movel(
                params=[prepick_point + DEFAULT_ORN],
                blocking=True, use_pos=True)
            self.ur5_pair.left_ur5.movel(
                params=[pick_point + DEFAULT_ORN],
                blocking=True, use_pos=True)
            self.ur5_pair.left_ur5.gripper.close(blocking=True)
            self.ur5_pair.left_ur5.movel(
                params=[backup_point + DEFAULT_ORN],
                j_vel=0.01, j_acc=0.01,
                blocking=True, use_pos=True)
            self.ur5_pair.left_ur5.movel(
                params=[prepick_point + DEFAULT_ORN],
                blocking=True, use_pos=True)
            self.ur5_pair.left_ur5.movel(
                params=[preplace_point + DEFAULT_ORN],
                blocking=True, use_pos=True)
            self.ur5_pair.left_ur5.movel(
                params=[place_point + DEFAULT_ORN],
                blocking=True, use_pos=True)
            self.ur5_pair.left_ur5.gripper.open(blocking=True)
            self.ur5_pair.left_ur5.movel(
                params=[preplace_point + DEFAULT_ORN],
                blocking=True, use_pos=True)
        elif arm == 'right':
            self.ur5_pair.right_ur5.movel(
                params=[prepick_point + DEFAULT_ORN],
                blocking=True, use_pos=True)
            self.ur5_pair.right_ur5.movel(
                params=[pick_point + DEFAULT_ORN],
                blocking=True, use_pos=True)
            self.ur5_pair.right_ur5.gripper.close(blocking=True)
            self.ur5_pair.right_ur5.movel(
                params=[backup_point + DEFAULT_ORN],
                j_vel=0.01, j_acc=0.01,
                blocking=True, use_pos=True)
            self.ur5_pair.right_ur5.movel(
                params=[prepick_point + DEFAULT_ORN],
                blocking=True, use_pos=True)
            self.ur5_pair.right_ur5.movel(
                params=[preplace_point + DEFAULT_ORN],
                blocking=True, use_pos=True)
            self.ur5_pair.right_ur5.movel(
                params=[place_point + DEFAULT_ORN],
                blocking=True, use_pos=True)
            self.ur5_pair.right_ur5.gripper.open(blocking=True)
            self.ur5_pair.right_ur5.movel(
                params=[preplace_point + DEFAULT_ORN],
                blocking=True, use_pos=True)
        # Lift up and see if cloth is stuck
        self.ur5_pair.move(
            move_type='l',
            params=[
                # left
                [0.5, 0.0, 0.0, *DEFAULT_ORN],
                # right
                [0.5, 0.0, 0.0, *DEFAULT_ORN]],
            blocking=True, use_pos=True)
        if should_grasp_cloth and self.compute_iou() > 0.75:
            raise GraspFailException
        self.ur5_pair.out_of_the_way()
        if self.dump_visualizations:
            self.stop_recording()

    def compute_coverage(self):
        rgb, depth = self.top_cam.get_rgbd()
        coverage = compute_coverage(rgb=rgb, depth=depth)
        print(
            f"\tCoverage: {coverage/CLOTHS_DATASET[CURRENT_CLOTH]['flatten_area']:.04f}")
        return coverage

    def get_obs(self):
        self._ensure_observing()
        self.raw_pretransform_rgb, \
            self.raw_pretransform_depth = self.top_cam.get_rgbd()
        self.raw_cloth_mask = self.get_cloth_mask(
            self.raw_pretransform_rgb.copy(), self.raw_pretransform_depth.copy())

        self.postcrop_pretransform_rgb = get_workspace_crop(
            self.raw_pretransform_rgb.copy())
        self.postcrop_pretransform_d = get_workspace_crop(
            self.raw_pretransform_depth.copy())
        # get_workspace_crop validates the square invariant required by the
        # policy and pixel-to-world mapping, with a calibration-specific error.

        self.pretransform_rgb = cv2.resize(
            self.postcrop_pretransform_rgb,
            (256, 256))
        self.pretransform_depth = cv2.resize(
            self.postcrop_pretransform_d,
            (256, 256))
        cloth_mask = cv2.resize(
            get_workspace_crop(self.raw_cloth_mask), (256, 256),
            interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        if self.replace_background:
            # Keep foreground=1 for all downstream geometry. Previously this
            # variable was replaced by its inverse, so adaptive scaling used
            # the background even though the displayed RGB looked masked.
            self.pretransform_rgb[cloth_mask == 0] = 0
        x, y = np.where(cloth_mask == 1)
        if not len(x):
            raise RuntimeError('cloth segmentation produced an empty mask; check the green segmentation overlay')
        dimx, dimy = self.pretransform_depth.shape
        minx = x.min()
        maxx = x.max()
        miny = y.min()
        maxy = y.max()

        self.adaptive_scale_factors = self.scale_factors.copy()
        if self.compute_coverage()/CLOTHS_DATASET[CURRENT_CLOTH]['flatten_area'] < 0.3:
            self.adaptive_scale_factors = self.adaptive_scale_factors[:4]
        if self.use_adaptive_scaling:
            try:
                # Minimum square crop
                cropx = max(dimx - 2*minx, dimx - 2*(dimx-maxx))
                cropy = max(dimy - 2*miny, dimy - 2*(dimy-maxy))
                crop = max(cropx, cropy)
                # Some breathing room
                crop = int(crop*1.5)
                if crop < dimx:
                    self.adaptive_scale_factors *= crop/dimx
                    self.episode_memory.add_value(
                        key='adaptive_scale',
                        value=float(crop/dimx))
            except Exception as e:
                print(e)
                print(self.current_task)
                exit()
        return preprocess_obs(
            self.pretransform_rgb.copy(),
            self.pretransform_depth.copy())

    def reset(self):
        self.episode_memory = Memory()
        self.episode_reward_sum = 0.
        self.current_timestep = -1
        self.terminate = False
        self.env_video_frames = {}
        if self.dump_visualizations:
            self.start_recording()
        self.current_task = self.get_task_fn()
        if self.dump_visualizations:
            self.stop_recording()
        self.current_timestep = 0
        self.init_coverage = self.compute_coverage()
        obs = self.get_obs()
        self.episode_memory.add_value(
            key='pretransform_observations', value=obs)
        self.episode_memory.add_value(
            key='failed_grasp', value=0)
        self.episode_memory.add_value(
            key='timed_out', value=0)
        self.episode_memory.add_value(
            key='cloth_stuck', value=0)
        self.transformed_obs = prepare_image(
            obs, self.get_transformations(), self.obs_dim,
            parallelize=self.parallelize_prepare_image)
        return self.transformed_obs, self.ray_handle

    def on_episode_end(self):
        self.stop_recording()
        super().on_episode_end(log=True)
        if os.path.exists(self.replay_buffer_path):
            with FileLock(self.replay_buffer_path + ".lock"):
                with h5py.File(self.replay_buffer_path, 'r') as file:
                    print('\tReplay Buffer Size:', len(file))
        print(
            '='*10 + f'EPISODE END IN {self.current_timestep} STEPS' + '='*10)

    def check_action_reachability(self, **kwargs):
        if os.environ.get('GEMSORT_ROBOTS', '1') != '1':
            return super().check_action_reachability(**kwargs)
        action = kwargs['action']
        if action not in ('fling', 'stretchdrag'):
            return False, f'GEMSORT action {action!r} is not implemented'
        left_pose = list(np.asarray(kwargs['p1'], float)) + list(DEFAULT_ORN)
        right_pose = list(np.asarray(kwargs['p2'], float)) + list(DEFAULT_ORN)
        self._show_grasp_candidate('checking 18-DOF reachability...')
        reachable, reason = self.ur5_pair.check_action_reachability(left_pose, right_pose)
        self._show_grasp_candidate('reachable' if reachable else f'rejected: {reason}')
        return reachable, reason

    def _show_grasp_candidate(self, status, pixels=None, world=None):
        """Refresh the raw-camera overlay even before a candidate is accepted."""
        from real_world import gemsort_viz
        if pixels is not None:
            self._candidate_pixels_raw = np.asarray(pixels, float)
        if world is not None:
            self._candidate_world = np.asarray(world, float)
        gemsort_viz.show(
            getattr(self, 'raw_pretransform_rgb', None),
            wait=1,
            status=f'step {self.current_timestep}: {status}',
            grasp_pixels=getattr(self, '_candidate_pixels_raw', None),
            grasp_world=getattr(self, '_candidate_world', None),
            segmentation_mask=getattr(self, 'raw_cloth_mask', None),
            policy_rgb=getattr(self, 'pretransform_rgb', None),
            policy_depth=getattr(self, 'pretransform_depth', None))

    def _show_grasp_verification(self, left_point, right_point):
        """With the grippers closed, show whether the fingertips actually landed on the cloth.

        The model is rendered from the pose the RealSense held during observation, and the garment segmented out of
        that same observation is composited back in. Same camera pose, same intrinsics, same resolution, so the two
        layers share a pixel grid and the comparison is direct rather than eyeballed across two windows.

        Diagnostic only: a failure here must never take down a run that is holding cloth.
        """
        from real_world import grasp_overlay, gemsort_viz
        try:
            rgb = getattr(self, 'raw_pretransform_rgb', None)
            if rgb is None:
                return
            height, width = rgb.shape[:2]
            intrinsics = self.top_cam.color_intr
            world_from_camera = self.top_cam_right_ur5_pose
            # MuJoCo's camera has its principal point at the image centre; the RealSense's need not be. A large
            # difference shifts the two layers by a constant offset, which would look exactly like a calibration
            # error in the fingertips. Say so rather than let it be read as one.
            offset = np.hypot(intrinsics[0, 2] - width / 2.0, intrinsics[1, 2] - height / 2.0)
            if offset > 15.0:
                print(f'[gemsort] NOTE: camera principal point is {offset:.0f} px off centre '
                      f'({intrinsics[0, 2]:.1f}, {intrinsics[1, 2]:.1f} vs {width/2:.0f}, {height/2:.0f}); the '
                      f'render is centred, so the sim layer is shifted by that much in this view.')
            sim = self.ur5_pair.render_from_camera(
                world_from_camera, grasp_overlay.fovy_degrees(intrinsics[1, 1], height), width, height)
            commanded = grasp_overlay.project([left_point, right_point], world_from_camera, intrinsics)
            frame = grasp_overlay.build(
                sim, rgb, mask=getattr(self, 'raw_cloth_mask', None), commanded_pixels=commanded,
                status=f'step {self.current_timestep}: grippers closed', max_width=1280)
            # Alongside the run's replay buffer, so the checks sit with the episode they belong to.
            buffer_path = getattr(self, 'replay_buffer_path', '')
            directory = os.path.dirname(buffer_path) if buffer_path else '.'
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, f'grasp_check_{strftime("%Y-%m-%d_%H-%M-%S")}.png')
            import cv2 as _cv2
            _cv2.imwrite(path, frame)
            print(f'[gemsort] grasp verification view written to {path}')
            module = gemsort_viz.cv2()
            if module is not None and gemsort_viz.levers()['enabled']:
                module.imshow('GEMSORT grasp check', frame)
                module.waitKey(1)
        except Exception as error:
            print(f'[gemsort] grasp verification view unavailable ({error})')

    def get_cam_pose(self):
        return self.top_cam_right_ur5_pose

    def check_action(self, **kwargs):
        try:
            retval = super().check_action(**kwargs)
        except Exception as error:
            # A transformed candidate can land on zero/invalid depth. It is
            # one candidate among many, not a reason to kill the policy loop.
            print(f'\tBad Grasp Candidate: {error}')
            return {'valid_action': False}
        p1, p2 = retval['pretransform_pixels'].copy()

        # need to convert from workspace pixels
        # to rectangular image pixels
        def process_pixels(pix):
            retval = pix.copy().astype(np.float32)
            ratio = self.postcrop_pretransform_d.shape[0] / \
                self.pretransform_depth.shape[0]
            retval *= ratio
            # Keep this signed. Invalid transformed candidates can be
            # negative; uint16 used to wrap them to ~65535 and then index the
            # raw camera arrays far out of bounds.
            retval = np.rint(retval).astype(np.int64)
            retval[0] += WS_PC[0]
            retval[1] += WS_PC[2]
            return retval
        p1 = process_pixels(p1)
        p2 = process_pixels(p2)

        # real world safety checks
        # if a grasp fails a safety check, then set
        # that action to invalid
        cam_intr = self.top_cam.color_intr
        if kwargs['action_primitive'] == 'fling':
            try:
                raw_h, raw_w = self.raw_pretransform_depth.shape
                # p1/p2 are (row, column); OpenCV wants (x, y). Draw the
                # policy estimate immediately, before geometry or planning can
                # reject it, so calibration failures remain visible.
                self._show_grasp_candidate('policy grasp candidate',
                                           pixels=np.asarray([p1[::-1], p2[::-1]]),
                                           world=[])
                if any(p.shape != (2,) or p[0] < 0 or p[0] >= raw_h or
                       p[1] < 0 or p[1] >= raw_w for p in (p1, p2)):
                    raise ValueError(f'grasp pixels outside {raw_w}x{raw_h} camera frame: {p1}, {p2}')
                # make sure they are far away from each other
                y, x = p1
                p1_grasp_cloth = bool(self.raw_cloth_mask[y, x])
                y2, x2 = p2
                p2_grasp_cloth = bool(self.raw_cloth_mask[y2, x2])
                if not p1_grasp_cloth or not p2_grasp_cloth:
                    raise Exception(f'grasp outside segmentation mask: p1={int(p1_grasp_cloth)}, '
                                    f'p2={int(p2_grasp_cloth)}')
                point_1 = list(pix_to_3d_position(
                    x=x, y=y,
                    depth_image=self.raw_pretransform_depth.copy(),
                    cam_intr=cam_intr,
                    cam_extr=self.top_cam_right_ur5_pose,
                    cam_depth_scale=self.cam_depth_scale))
                y, x = p2
                point_2 = list(pix_to_3d_position(
                    x=x, y=y,
                    depth_image=self.raw_pretransform_depth.copy(),
                    cam_intr=cam_intr,
                    cam_extr=self.top_cam_right_ur5_pose,
                    cam_depth_scale=self.cam_depth_scale))
                grasp_width = np.linalg.norm(
                    np.array(point_1) - np.array(point_2))
                min_grasp_width = float(os.environ.get('GEMSORT_MIN_GRASP_WIDTH_M', MIN_GRASP_WIDTH))
                max_grasp_width = float(os.environ.get('GEMSORT_MAX_GRASP_WIDTH_M', MAX_GRASP_WIDTH))
                if not 0 <= min_grasp_width < max_grasp_width:
                    raise ValueError(f'invalid grasp-width limits {min_grasp_width}..{max_grasp_width} m')
                if grasp_width < min_grasp_width:
                    raise Exception(
                        f'Grasp width too small: {grasp_width:.03f} < {min_grasp_width:.03f} m')
                if grasp_width > max_grasp_width:
                    raise Exception(
                        f'Grasp width too large: {grasp_width:.03f} > {max_grasp_width:.03f} m')
                if point_1[0] < point_2[0]:
                    # point 1 is to the right of point 2
                    left_point = list(pix_to_3d_position(
                        x=p2[1], y=p2[0],
                        depth_image=self.raw_pretransform_depth.copy(),
                        cam_intr=cam_intr,
                        cam_extr=self.top_cam_left_ur5_pose,
                        cam_depth_scale=self.cam_depth_scale))
                    right_point = point_1
                    left_grasp_cloth = p2_grasp_cloth
                    right_grasp_cloth = p1_grasp_cloth
                else:
                    # point 2 is to the right of point 1
                    left_point = list(pix_to_3d_position(
                        x=p1[1], y=p1[0],
                        depth_image=self.raw_pretransform_depth.copy(),
                        cam_intr=cam_intr,
                        cam_extr=self.top_cam_left_ur5_pose,
                        cam_depth_scale=self.cam_depth_scale))
                    right_point = point_2
                    left_grasp_cloth = p1_grasp_cloth
                    right_grasp_cloth = p2_grasp_cloth
                self._show_grasp_candidate('camera points reconstructed',
                                           world=[left_point, right_point])
                # Sanity bound in the CAGE WORLD frame the reconstruction produces (table top at
                # +WORKSPACE_SURFACE). Upstream compared against 0.0 because its UR5 frame put the table at
                # -0.15; that literal silently passes every cage-frame point, including ones behind the camera.
                ceiling = WORKSPACE_SURFACE + MAX_GRASP_HEIGHT_ABOVE_SURFACE
                if right_point[2] > ceiling or left_point[2] > ceiling:
                    raise Exception(
                        f'Grasp points more than {MAX_GRASP_HEIGHT_ABOVE_SURFACE:.2f} m above the '
                        f'{WORKSPACE_SURFACE:.2f} m table top, probably a calibration error: ' +
                        ','.join(np.array(right_point).astype(str)) + '|' +
                        ','.join(np.array(left_point).astype(str)))
                floor = WORKSPACE_SURFACE - MAX_GRASP_HEIGHT_ABOVE_SURFACE
                if right_point[2] < floor or left_point[2] < floor:
                    raise Exception(
                        f'Grasp points below the {WORKSPACE_SURFACE:.2f} m table top, probably a '
                        f'calibration error: ' +
                        ','.join(np.array(right_point).astype(str)) + '|' +
                        ','.join(np.array(left_point).astype(str)))
                retval.update({
                    'valid_action': True,
                    'p1': left_point,
                    'p2': right_point,
                    'grasp_width': grasp_width,
                    'p1_grasp_cloth': left_grasp_cloth,
                    'p2_grasp_cloth': right_grasp_cloth
                })
                return retval
            except Exception as e:
                self._show_grasp_candidate(f'rejected: {e}')
                print('\tBad Grasp Candidate:', e)
                return {'valid_action': False}
        # if not a fling then can just return
        return retval
