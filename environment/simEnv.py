from .Memory import Memory
import numpy as np
from .utils import (
    preprocess_obs,
    blender_render_cloth,
    get_cloth_mesh,
    visualize_action,
    compute_pose,
    get_largest_component,
    pixels_to_3d_positions)
import torch
from .exceptions import MoveJointsException
from learning.nets import prepare_image
from typing import List, Callable
from itertools import product
from environment.flex_utils import (
    set_scene,
    get_image,
    get_current_covered_area,
    wait_until_stable,
    PickerPickPlace)
from .tasks import Task
from tqdm import tqdm
from time import time
import hashlib
import imageio
import os
import pyflex
import cv2


class SimEnv:
    def __init__(self,
                 replay_buffer_path: str,
                 obs_dim: int,
                 num_rotations: int,
                 scale_factors: List[float],
                 get_task_fn: Callable[[None], Task],
                 action_primitives: List[str],
                 pix_grasp_dist: int,
                 pix_drag_dist: int,
                 pix_place_dist: int,
                 stretchdrag_dist: float,
                 reach_distance_limit: float,
                 fixed_fling_height: float,
                 conservative_grasp_radius: int = 4,
                 use_adaptive_scaling=True,
                 dump_visualizations=False,
                 parallelize_prepare_image=False,
                 gui=False,
                 grasp_height=0.02,
                 fling_speed=6e-3,
                 episode_length=10,
                 render_dim=400,
                 particle_radius=0.00625,
                 render_engine='opengl',
                 **kwargs):
        # environment state variables
        self.grasp_states = [False, False]
        self.ray_handle = None
        self.particle_radius = particle_radius
        self.replay_buffer_path = replay_buffer_path
        self.log_dir = os.path.dirname(self.replay_buffer_path)
        self.image_dim = render_dim  # what to render blender with
        self.obs_dim = obs_dim  # what to resize to fit in net
        self.episode_length = episode_length
        self.render_engine = render_engine

        self.conservative_grasp_radius = conservative_grasp_radius
        self.rotations = [(2*i/(num_rotations-1) - 1) * 90
                          for i in range(num_rotations)]
        if 'fling' not in action_primitives:
            # When using pick & place or pick & drag, allow model
            # to place all 360 degrees around pick
            self.rotations = [(2*i/num_rotations - 1) *
                              180 for i in range(num_rotations)]
        self.scale_factors = np.array(scale_factors)
        self.use_adaptive_scaling = use_adaptive_scaling
        self.adaptive_scale_factors = self.scale_factors.copy()

        # primitives parameters
        self.grasp_height = grasp_height
        self.pix_grasp_dist = pix_grasp_dist
        self.pix_drag_dist = pix_drag_dist
        self.pix_place_dist = pix_place_dist
        self.stretchdrag_dist = stretchdrag_dist
        self.fling_speed = fling_speed
        self.default_speed = 1e-2
        self.fixed_fling_height = fixed_fling_height

        # visualizations
        self.dump_visualizations = dump_visualizations
        self.parallelize_prepare_image = parallelize_prepare_image
        if gui:
            self.parallelize_prepare_image = True
        self.gui = gui
        self.gif_freq = 24
        self.env_video_frames = {}

        # physical limit of dual arm system
        self.left_arm_base = np.array([0.765, 0, 0])
        self.right_arm_base = np.array([-0.765, 0, 0])
        self.reach_distance_limit = reach_distance_limit

        # tasks
        self.current_task = None
        self.get_task_fn = get_task_fn
        self.gui_render_freq = 2
        self.gui_step = 0
        self.setup_env()
        self.action_handlers = {
            'fling': self.pick_and_fling_primitive,
            'stretchdrag': self.pick_stretch_drag_primitive,
            'drag': self.pick_and_drag_primitive,
            'place': self.pick_and_place_primitive
        }

    def step_simulation(self):
        pyflex.step()
        if self.gui and self.gui_step % self.gui_render_freq == 0:
            pyflex.render()
        self.gui_step += 1

    def setup_env(self):
        pyflex.init(
            not self.gui,  # headless: bool.
            True,      # render: bool
            720, 720)  # camera dimensions: int x int
        self.action_tool = PickerPickPlace(
            num_picker=2,
            particle_radius=self.particle_radius,
            picker_radius=self.grasp_height,
            picker_low=(-5, 0, -5),
            picker_high=(5, 5, 5))

    def get_transformations(self):
        return list(product(
            self.rotations, self.adaptive_scale_factors))

    def stretch_cloth(self,
                      grasp_dist: float,
                      fling_height: float = 0.7,
                      max_grasp_dist: float = 0.7,
                      increment_step=0.02):
        # keep stretching until cloth is tight
        # i.e.: the midpoint of the grasped region
        # stops moving
        left, right = self.action_tool._get_pos()[0]
        left[1] = fling_height
        right[1] = fling_height
        midpoint = (left + right)/2
        direction = left - right
        direction = direction/np.linalg.norm(direction)
        self.movep([left, right], speed=5e-4, min_steps=20)
        stable_steps = 0
        cloth_midpoint = 1e2
        while True:
            positions = pyflex.get_positions().reshape((-1, 4))[:, :3]
            # get midpoints
            high_positions = positions[positions[:, 1] > fling_height-0.1, ...]
            if (high_positions[:, 0] < 0).all() or \
                    (high_positions[:, 0] > 0).all():
                # single grasp
                return grasp_dist
            positions = [p for p in positions]
            positions.sort(
                key=lambda pos: np.linalg.norm(pos[[0, 2]]-midpoint[[0, 2]]))
            new_cloth_midpoint = positions[0]
            stable = np.linalg.norm(
                new_cloth_midpoint - cloth_midpoint) < 1.5e-2
            if stable:
                stable_steps += 1
            else:
                stable_steps = 0
            stretched = stable_steps > 2
            if stretched:
                return grasp_dist
            cloth_midpoint = new_cloth_midpoint
            grasp_dist += increment_step
            left = midpoint + direction*grasp_dist/2
            right = midpoint - direction*grasp_dist/2
            self.movep([left, right], speed=5e-4)
            if grasp_dist > max_grasp_dist:
                return max_grasp_dist

    def lift_cloth(self,
                   grasp_dist: float,
                   fling_height: float = 0.7,
                   increment_step: float = 0.05,
                   max_height=0.7):
        while True:
            positions = pyflex.get_positions().reshape((-1, 4))[:, :3]
            heights = positions[:, 1]
            if heights.min() > 0.02:
                return fling_height
            fling_height += increment_step
            self.movep([[grasp_dist/2, fling_height, -0.3],
                        [-grasp_dist/2, fling_height, -0.3]], speed=1e-3)
            if fling_height >= max_height:
                return fling_height

    def check_action(self, action_primitive, pixels,
                     transformed_depth, transformed_rgb,
                     scale, rotation,
                     value_map=None, all_value_maps=None,
                     **kwargs):
        if not hasattr(self, "_radius_printed"):        #Test ob Strenger Candidate check
            print("conservative_grasp_radius:",
                  self.conservative_grasp_radius)
            self._radius_printed = True
        args = {
            'pretransform_depth': self.pretransform_depth.copy(),
            'pretransform_rgb': self.pretransform_rgb.copy(),
            'transformed_depth': transformed_depth.copy(),
            'transformed_rgb': transformed_rgb.copy(),
            'scale': scale,
            'rotation': rotation
        }
        retval = pixels_to_3d_positions(
            pixels=pixels,
            pose_matrix=compute_pose(
                pos=[0, 2, 0],
                lookat=[0, 0, 0],
                up=[0, 0, 1]), **args)
        #------------TEST für achsen---------------------------
        #print("\n========== VISUALIZATION DEBUG ==========")
        #print("ROTATION:", rotation)
        #print("SCALE:", scale)
        #print("TRANSFORMED PIXELS:", pixels)
        #print("PRETRANSFORM PIXELS:", retval['pretransform_pixels'])
        #print("\n========== RETVAL PRETRANSFORM PIXELS======= def check_action")
        #print(retval["pretransform_pixels"])
        pix1, pix2 = retval['pretransform_pixels']      #Test Pixel bis Print out of bounds
        for name, pix in [("P1", pix1), ("P2", pix2)]:

            if (0 <= pix[0] < self.pretransform_depth.shape[0] and
                    0 <= pix[1] < self.pretransform_depth.shape[1]):

                d = self.pretransform_depth[pix[0], pix[1]]

                #print(f"DEPTH400 {name}:", d)
                #print(f"MASK400 {name}:", d < 1.99)

            else:
                pass
                #print(f"{name} OUT OF BOUNDS:", pix)


        def get_action_visualization():
            return visualize_action(
                action_primitive=action_primitive,
                transformed_pixels=pixels,
                pretransform_pixels=retval['pretransform_pixels'],
                value_map=value_map,
                all_value_maps=all_value_maps,
                **args)

        retval.update({
            'get_action_visualization_fn': get_action_visualization
        })

        #cloth_mask = (self.pretransform_depth != 2.0).astype(np.uint8)  #OG

        # ============================================
        # DEBUG: RGB vs DEPTH MASK OVERLAY
        # ============================================
        cloth_mask = (self.pretransform_depth < 1.99).astype(np.uint8)  # Test
        vals = np.unique(np.round(self.pretransform_depth, 3))
        #print("DEPTH VALUES:", vals[-20:])

        #print("DEPTH MIN:", self.pretransform_depth.min())
        #print("DEPTH MAX:", self.pretransform_depth.max())

        #print("CLOTH PIXELS:", cloth_mask.sum(), "/", cloth_mask.size)

        #print("DEPTH[0,0]:", self.pretransform_depth[0, 0])
        #---------- MATLAB PLOT CODEABSCHNITT-------
        #import cv2

        #cv2.imwrite(
        #    "cloth_mask.png",
        #    cloth_mask * 255
        #)

        #import matplotlib.pyplot as plt

        #plt.figure(figsize=(6, 6))
        #plt.imshow(self.pretransform_rgb)

        #mask_vis = np.zeros((*cloth_mask.shape, 4))
        #mask_vis[..., 1] = cloth_mask * 1.0  # grün
        #mask_vis[..., 3] = cloth_mask * 0.3  # transparent

        #plt.imshow(mask_vis)
        #pix_1, pix_2 = retval['pretransform_pixels']

        #plt.scatter(
        #    pix_1[1], pix_1[0],
        #    c='red',
        #    s=100
        #)

        #plt.scatter(
        #    pix_2[1], pix_2[0],
        #    c='blue',
        #    s=100
        #)

        #plt.title("RGB + CLOTH MASK + GRASP POINTS")
        #plt.savefig("mask_debug.png")
        #plt.close()
        #-------ENDE MATLAB CODEABSCHNITT--------

        # ============================================
        # ORIGINAL CODE
        # ============================================
        pix_1, pix_2 = retval['pretransform_pixels']
        if self.conservative_grasp_radius > 0:
            grasp_mask_1 = np.zeros(cloth_mask.shape)
            grasp_mask_1 = cv2.circle(
                img=grasp_mask_1,
                center=(pix_1[1], pix_1[0]),
                radius=self.conservative_grasp_radius,
                color=1, thickness=-1).astype(bool)
            grasp_mask_2 = np.zeros(cloth_mask.shape)
            grasp_mask_2 = cv2.circle(
                img=grasp_mask_2,
                center=(pix_2[1], pix_2[0]),
                radius=self.conservative_grasp_radius,
                color=1, thickness=-1).astype(bool)
            retval.update({
                'p1_grasp_cloth': cloth_mask[grasp_mask_1].all(),
                'p2_grasp_cloth': cloth_mask[grasp_mask_2].all(),
            })
            #TEST Grasp auf Cloth
            #print("--------GRASP LOG--------")
            #print("P1_GRASP_CLOTH:", retval['p1_grasp_cloth'])
            #print("P2_GRASP_CLOTH:", retval['p2_grasp_cloth'])
            #Test tr.mat 4 prints pix related
            #print("PIX_1:", pix_1)
            #print("PIX_2:", pix_2)

            #print(
            #    "DEPTH PIX_1:",
            #    self.pretransform_depth[pix_1[0], pix_1[1]],
            #    "MASK:",
            #    cloth_mask[pix_1[0], pix_1[1]]
            #)
            #print(
            #    "DEPTH PIX_1 SWAPPED:",
            #    self.pretransform_depth[pix_1[1], pix_1[0]]
            #)
            #print(
            #    "DEPTH PIX_2:",
            #    self.pretransform_depth[pix_2[0], pix_2[1]],
            #    "MASK:",
            #    cloth_mask[pix_2[0], pix_2[1]]
            #)
            #print(
            #    "DEPTH PIX_2 SWAPPED:",
            #    self.pretransform_depth[pix_2[1], pix_2[0]]
            #)
            #print("PRETRANSFORM PIXELS:", retval['pretransform_pixels'])
            #print("conservative_grasp_radius:", self.conservative_grasp_radius)
            #print("\n--- MASK DEBUG ---")
            #print("p1:", pix_1)
            #print("p2:", pix_2)

            #print("cloth_mask shape:", cloth_mask.shape)

            #print("grasp_mask_1 sum:", grasp_mask_1.sum())
            #print("grasp_mask_2 sum:", grasp_mask_2.sum())

            #print("grasp_mask_1 shape:", grasp_mask_1.shape)
            #print("grasp_mask_2 shape:", grasp_mask_2.shape)
            #print("cloth pixels in grasp mask 1:", cloth_mask[grasp_mask_1].sum(), "/", grasp_mask_1.sum())
            #print("cloth pixels in grasp mask 2:", cloth_mask[grasp_mask_2].sum(), "/", grasp_mask_2.sum())

        else:
            retval.update({
                'p1_grasp_cloth': True,
                'p2_grasp_cloth': True,
            })
        retval['valid_action'] = (          #test tr.mat
                retval['valid_action']
                and retval['p1_grasp_cloth']
                and retval['p2_grasp_cloth']
        )
        # TODO can probably refactor so args to primitives have better variable names
        #print("\n===== FINAL ACTION CHECK =====")
        #print("valid_action before:", retval['valid_action'])
        #print("p1_grasp_cloth:", retval['p1_grasp_cloth'])
        #print("p2_grasp_cloth:", retval['p2_grasp_cloth'])
        return retval

    def fling_primitive(self, dist, fling_height, fling_speed):
        # fling
        self.movep([[dist/2, fling_height, -0.2],
                    [-dist/2, fling_height, -0.2]], speed=fling_speed)
        self.movep([[dist/2, fling_height, 0.2],
                    [-dist/2, fling_height, 0.2]], speed=fling_speed)
        self.movep([[dist/2, fling_height, 0.2],
                    [-dist/2, fling_height, 0.2]], speed=1e-2, min_steps=4)
        # lower
        self.movep([[dist/2, self.grasp_height*2, -0.2],
                    [-dist/2, self.grasp_height*2, -0.2]], speed=1e-2)
        self.movep([[dist/2, self.grasp_height*2, -0.25],
                    [-dist/2, self.grasp_height*2, -0.25]], speed=5e-3)
        # release
        self.set_grasp(False)
        if self.dump_visualizations:
            self.movep(
                [[dist/2, self.grasp_height*2, -0.25],
                 [-dist/2, self.grasp_height*2, -0.25]], min_steps=10)
        self.reset_end_effectors()

    def pick_and_fling_primitive(
            self, p1, p2,
            p1_grasp_cloth: bool,
            p2_grasp_cloth: bool):
        if not (p1_grasp_cloth or p2_grasp_cloth):
            # both points not on cloth
            return
        left_grasp_pos, right_grasp_pos = p1, p2
        left_grasp_pos[1] = self.grasp_height
        right_grasp_pos[1] = self.grasp_height

        # grasp distance
        dist = np.linalg.norm(
            np.array(left_grasp_pos) - np.array(right_grasp_pos))
        self.movep([left_grasp_pos, right_grasp_pos])
        # only grasp points on cloth
        self.grasp_states = [p1_grasp_cloth, p2_grasp_cloth]
        if self.dump_visualizations:
            self.movep([left_grasp_pos, right_grasp_pos], min_steps=10)

        # lift to prefling
        self.movep([[dist/2, 0.3, -0.3], [-dist/2, 0.3, -0.3]], speed=5e-3)
        print("\n===== CLOTH GRASP CHECK =====")    #Test um zu sehen ob Cloth wirlklich erreicht wird
        print("grasp_states:", self.grasp_states)
        print("dist:", dist)
        #if not self.is_cloth_grasped():      #OG bis return (3Z)
        #    self.terminate = True
        #    return
        grasped = self.is_cloth_grasped()       #Test bis if not grasped:   return

        print("is_cloth_grasped:", grasped)

        if not grasped:
            print("FLING ABORTED")
            self.terminate = True
            return
        dist = self.stretch_cloth(grasp_dist=dist, fling_height=0.3)

        if self.fixed_fling_height == -1:
            fling_height = self.lift_cloth(
                grasp_dist=dist, fling_height=0.3)
        else:
            fling_height = self.fixed_fling_height
        self.fling_primitive(
            dist=dist,
            fling_height=fling_height,
            fling_speed=self.fling_speed)

    def pick_and_drag_primitive(
            self, p1, p2,
            p1_grasp_cloth: bool,
            p2_grasp_cloth: bool):
        if not p1_grasp_cloth:
            # First grasp point not on cloth
            return
        # prepare primitive params
        start_drag_pos, end_drag_pos = p1, p2
        start_drag_pos[1] = self.grasp_height
        end_drag_pos[1] = self.grasp_height

        prestart_drag_pos = start_drag_pos.copy()
        prestart_drag_pos[1] = 0.3
        postend_drag_pos = end_drag_pos.copy()
        postend_drag_pos[1] = 0.3

        # execute action
        self.movep([prestart_drag_pos, [-0.2, 0.3, -0.2]], speed=5e-3)
        self.movep([start_drag_pos, [-0.2, 0.3, -0.2]], speed=5e-3)
        self.set_grasp(True)
        self.movep([end_drag_pos, [-0.2, 0.3, -0.2]], speed=5e-3)
        self.set_grasp(False)
        self.movep([postend_drag_pos, [-0.2, 0.3, -0.2]], speed=5e-3)
        self.reset_end_effectors()

    def pick_and_place_primitive(
        self, p1, p2,
        p1_grasp_cloth: bool, p2_grasp_cloth: bool,
            lift_height=0.2):
        if not p1_grasp_cloth:
            # First grasp point not on cloth
            return
        # prepare primitive params
        pick_pos, place_pos = p1, p2
        pick_pos[1] = self.grasp_height
        place_pos[1] = self.grasp_height

        prepick_pos = pick_pos.copy()
        prepick_pos[1] = lift_height
        preplace_pos = place_pos.copy()
        preplace_pos[1] = lift_height

        # execute action
        self.movep([prepick_pos, [-0.2, 0.3, -0.2]], speed=5e-3)
        self.movep([pick_pos, [-0.2, 0.3, -0.2]], speed=5e-3)
        self.set_grasp(True)
        self.movep([prepick_pos, [-0.2, 0.3, -0.2]], speed=5e-3)
        self.movep([preplace_pos, [-0.2, 0.3, -0.2]], speed=5e-3)
        self.movep([place_pos, [-0.2, 0.3, -0.2]], speed=5e-3)
        self.set_grasp(False)
        self.movep([preplace_pos, [-0.2, 0.3, -0.2]], speed=5e-3)
        self.reset_end_effectors()

    def pick_stretch_drag_primitive(
            self, p1, p2,
            p1_grasp_cloth: bool,
            p2_grasp_cloth: bool):
        if not (p1_grasp_cloth or p2_grasp_cloth):
            # both points not on cloth
            return
        left_start_drag_pos, right_start_drag_pos = p1, p2
        left_start_drag_pos[1] = self.grasp_height
        right_start_drag_pos[1] = self.grasp_height

        left_prestart_drag_pos = left_start_drag_pos.copy()
        left_prestart_drag_pos[1] = 0.3
        right_prestart_drag_pos = right_start_drag_pos.copy()
        right_prestart_drag_pos[1] = 0.3

        self.movep([left_prestart_drag_pos, right_prestart_drag_pos])
        self.movep([left_start_drag_pos, right_start_drag_pos], speed=2e-3)
        # only grasp points on cloth
        self.set_grasp([p1_grasp_cloth,
                        p2_grasp_cloth])
        if self.dump_visualizations:
            self.movep(
                [left_start_drag_pos, right_start_drag_pos], min_steps=10)

        # grasp distance
        dist = np.linalg.norm(
            np.array(left_start_drag_pos) - np.array(right_start_drag_pos))

        # stretch if cloth is grasped by both
        if all(self.grasp_states):
            dist = self.stretch_cloth(
                grasp_dist=dist, fling_height=self.grasp_height)

        # compute drag direction
        drag_direction = np.cross(
            left_start_drag_pos - right_start_drag_pos, np.array([0, 1, 0]))
        drag_direction = self.stretchdrag_dist * \
            drag_direction / np.linalg.norm(drag_direction)
        left_start_drag_pos, right_start_drag_pos = \
            self.action_tool._get_pos()[0]
        left_end_drag_pos = left_start_drag_pos + drag_direction
        right_end_drag_pos = right_start_drag_pos + drag_direction
        # prevent ee go under cloth
        left_end_drag_pos[1] += 0.1
        right_end_drag_pos[1] += 0.1

        left_postend_drag_pos = left_end_drag_pos.copy()
        left_postend_drag_pos[1] = 0.3
        right_postend_drag_pos = right_end_drag_pos.copy()
        right_postend_drag_pos[1] = 0.3

        self.movep([left_end_drag_pos, right_end_drag_pos], speed=2e-3)
        self.set_grasp(False)
        self.movep([left_postend_drag_pos, right_postend_drag_pos])
        self.reset_end_effectors()

    def compute_coverage(self):
        return get_current_covered_area(self.particle_radius)

    def log_step_stats(self, action):
        self.episode_memory.add_observation(action['observation'])
        self.episode_memory.add_action(action['action_mask'])
        self.episode_memory.add_value(
            key='action_visualization',
            value=action['action_visualization'])
        self.episode_memory.add_value(
            key='rotation', value=float(action['rotation']))
        self.episode_memory.add_value(
            key='scale', value=float(action['scale']))
        self.episode_memory.add_value(
            key='value_map',
            value=action['value_map'])
        self.episode_memory.add_value(
            key='action_primitive',
            value=action['action_primitive'])
        self.episode_memory.add_value(
            key='max_indices',
            value=np.array(action['max_indices']))
        for key, value in self.current_task.get_stats().items():
            self.episode_memory.add_value(
                key=key,
                value=value)
        if self.dump_visualizations:
            if action['all_value_maps'] is not None:
                self.episode_memory.add_value(
                    key='value_maps',
                    value=action['all_value_maps'])
            self.episode_memory.add_value(
                key='all_obs',
                value=self.transformed_obs)

    def preaction(self):
        self.preaction_positions = pyflex.get_positions().reshape(-1, 4)[:, :3]

    def postaction(self):
        self.reset_end_effectors()
        wait_until_stable(gui=self.gui)
        postaction_positions = pyflex.get_positions().reshape(-1, 4)[:, :3]
        deltas = np.linalg.norm(
            np.abs(postaction_positions - self.preaction_positions), axis=1)
        if deltas.max() < 5e-2:
            # if didn't really move cloth then end early
            self.terminate = True

    def step(self, value_maps):
        # Log stats before perform actions
        self.preaction()

        prev_coverage = self.compute_coverage()
        self.episode_memory.add_value(
            key='preaction_coverage',
            value=float(prev_coverage))
        action_primitive, action = self.get_max_value_valid_action(value_maps)
        if action_primitive is not None and action is not None:
            self.action_handlers[action_primitive](**action)
        self.postaction()

        # Log stats after perform actions
        curr_coverage = self.compute_coverage()
        self.episode_memory.add_value(
            key='postaction_coverage',
            value=float(curr_coverage))

        self.current_timestep += 1
        self.terminate = self.terminate or \
            self.current_timestep >= self.episode_length
        self.episode_memory.add_rewards_and_termination(
            curr_coverage - prev_coverage, self.terminate)
        obs = self.get_obs()
        self.episode_memory.add_value(
            key='next_observations', value=obs)
        if self.terminate:
            self.on_episode_end()
            return self.reset()
        else:
            self.episode_memory.add_value(
                key='pretransform_observations', value=obs)
        self.transformed_obs = prepare_image(
            obs, self.get_transformations(), self.obs_dim,
            parallelize=self.parallelize_prepare_image)
        return self.transformed_obs, self.ray_handle

    def get_action_params(self, action_primitive, max_indices):
        x, y, z = max_indices
        if action_primitive == 'fling' or\
                action_primitive == 'stretchdrag':
            center = np.array([x, y, z])
            p1 = center[1:].copy()
            p1[0] = p1[0] + self.pix_grasp_dist
            p2 = center[1:].copy()
            p2[0] = p2[0] - self.pix_grasp_dist
        elif action_primitive == 'drag':
            p1 = np.array([y, z])
            p2 = p1.copy()
            p2[0] += self.pix_drag_dist
        elif action_primitive == 'place':
            p1 = np.array([y, z])
            p2 = p1.copy()
            p2[0] += self.pix_place_dist
        else:
            raise Exception(
                f'Action Primitive not supported: {action_primitive}')
        return p1, p2

    def check_arm_reachability(self, arm_base, reach_pos):
        #print("REACH LIMIT:", self.reach_distance_limit)    #Test
        return np.linalg.norm(arm_base - reach_pos) < self.reach_distance_limit

    def check_action_reachability(
            self, action: str, p1: np.array, p2: np.array):
        #print("\n=== REACHABILITY DEBUG ===")       #TestBlock bis if action
        #print("ACTION:", action)

        #print("P1 WORLD:", p1)
        #print("P2 WORLD:", p2)

        #print("LEFT BASE:", self.left_arm_base)
        #print("RIGHT BASE:", self.right_arm_base)

        #print("DIST LEFT->P1:",
        #      np.linalg.norm(p1 - self.left_arm_base))

        #print("DIST RIGHT->P2:",
        #      np.linalg.norm(p2 - self.right_arm_base))
        if action == 'fling' or action == 'stretchdrag':
            # right and left must reach each point respectively
            return self.check_arm_reachability(self.left_arm_base, p1) \
                and self.check_arm_reachability(self.right_arm_base, p2), None
        elif action == 'drag' or action == 'place':
            # either right can reach both or left can reach both
            if self.check_arm_reachability(self.left_arm_base, p1) and\
                    self.check_arm_reachability(self.left_arm_base, p2):
                return True, 'left'
            elif self.check_arm_reachability(self.right_arm_base, p1) and \
                    self.check_arm_reachability(self.right_arm_base, p2):
                return True, 'right'
            else:
                return False, None
        raise NotImplementedError()

    def get_max_value_valid_action(self, value_maps) -> dict:
        stacked_value_maps = torch.stack(tuple(value_maps.values()))
        print("STACKED SHAPE:", stacked_value_maps.shape)       #Test hoher Count
        print("TOTAL ENTRIES:", stacked_value_maps.numel())     #Test hoher Count

        # (**) filter out points too close to edge
        stacked_value_maps = stacked_value_maps[
            :, :,
            self.pix_grasp_dist:-self.pix_grasp_dist,
            self.pix_grasp_dist:-self.pix_grasp_dist]

        # TODO make more efficient by creating index list,
        # flattened value list, then sort and eliminate
        sorted_values, _ = stacked_value_maps.flatten().sort(descending=True)
        actions = list(value_maps.keys())
        #print("\n===== VALUE MAP KEYS =====")
        #print(list(value_maps.keys()))
        print("\n===== MAX VALUE PER PRIMITIVE =====")  #Test primitive choice
        for a in actions:
            print(
                f"{a:12s}",
                value_maps[a].max().item()
            )
        candidate_counter = 0       #Test Counter
        invalid_counter = 0         #Test Counter
        self.best_p1_ratio = 0.0    #Test hoher Counter
        self.best_p2_ratio = 0.0    #Test hoher Counter
        for value in sorted_values:
            #print("\n===================================") #Test
            #print("TESTING NEW VALUE CANDIDATE")    #Test
            #print("VALUE:", value.item())   #Test

            for indices in np.array(np.where(stacked_value_maps == value)).T:
                candidate_counter += 1      #Test Counter
                if candidate_counter % 3000 == 0:       #Test Counter alle 3000#1000
                    print("Candidates tested:", candidate_counter)
                # Account for index of filtered pixels. See (**) above
                indices[-2:] += self.pix_grasp_dist

                max_indices = indices[1:]
                x, y, z = max_indices
                action = actions[indices[0]]
                #print("\nACTION:", action)  #Test
                #print("INDICES:", indices)  #Test
                value_map = value_maps[action]
                #print("VALUE MAP SHAPE:", value_map.shape)  # Test bis tuple(value_map.shape))
                #print("BEST VALUE:", value_map.max().item())
                flat_idx = value_map.argmax().item()
                best_idx = np.unravel_index(
                    flat_idx,
                    tuple(value_map.shape)
                )
                #print("BEST INDEX:", best_idx)
                reach_points = np.array(self.get_action_params(
                    action_primitive=action,
                    max_indices=(x, y, z)))
                # if any point is outside domain, skip
                if any(((p < 0).any() or (p >= self.obs_dim).any())
                       for p in reach_points):
                    continue
                p1, p2 = reach_points[:2]
                #print("NETWORK P1:", p1)    #Test
                #print("NETWORK P2:", p2)    #Test
                transformed_depth_64 = self.transformed_obs[x, 3, :, :].numpy() #Test
                cloth_mask_64 = transformed_depth_64< 1.99 #Test
                #print("NETWORK P1 ON CLOTH:", cloth_mask_64[p1[1], p1[0]]) #Test
                #print("NETWORK P2 ON CLOTH:", cloth_mask_64[p2[1], p2[0]]) #Test
                #print("DEPTH64 P1:", transformed_depth_64[p1[1], p1[0]])    #Test
                #print("DEPTH64 P2:",transformed_depth_64[p2[1], p2[0]])     #Test
                #print("NUM CLOTH PIXELS:", (transformed_depth_64 < 1.99).sum()) #Test
                #print("DEPTH MIN:", transformed_depth_64.min())             #Test
                #print("DEPTH MAX:", transformed_depth_64.max())             #Test
                img64 = transformed_depth_64.copy()
                #plt.imshow(cloth_mask_64)                                   #Test
                #plt.scatter(p1[0], p1[1], c='red')                          #Test
                #plt.scatter(p2[0], p2[1], c='blue')                         #Test
                #plt.show()                                                  #Test
                action_mask = torch.zeros(value_map.size()[1:])
                action_mask[y, z] = 1
                num_scales = len(self.adaptive_scale_factors)
                rotation_idx = x // num_scales
                scale_idx = x - rotation_idx * num_scales
                scale = self.adaptive_scale_factors[scale_idx]
                rotation = self.rotations[rotation_idx]
                action_kwargs = {
                    'observation': self.transformed_obs[x, ...],
                    'action_primitive': action,
                    'p1': p1,
                    'p2': p2,
                    'scale': scale,
                    'rotation': rotation,
                    'max_indices': max_indices,
                    'action_mask': action_mask,
                    'value_map': value_map[x, :, :],
                    'all_value_maps': value_map,
                    'info': None
                }
                action_kwargs.update({
                    'transformed_depth':
                    action_kwargs['observation'][3, :, :].numpy(),
                    'transformed_rgb':
                    action_kwargs['observation'][:3, :, :].numpy(),
                })
                action_params = self.check_action(
                    pixels=np.array([p1, p2]),
                    **action_kwargs)
                if not action_params['valid_action']:
                    invalid_counter += 1      #Test Counter
                    continue
                reachable, left_or_right = self.check_action_reachability(
                    action=action,
                    p1=action_params['p1'],
                    p2=action_params['p2'])
                #print("reachable:", reachable)  #Test
                if action == 'place' or action == 'drag':
                    action_kwargs['left_or_right'] = left_or_right

                if action == 'stretchdrag':
                    left_start_drag_pos = action_params['p1']
                    right_start_drag_pos = action_params['p2']
                    left_start_drag_pos[1] = self.grasp_height
                    right_start_drag_pos[1] = self.grasp_height

                    # compute drag direction
                    drag_direction = np.cross(
                        left_start_drag_pos - right_start_drag_pos,
                        np.array([0, 1, 0]))
                    drag_direction = self.stretchdrag_dist * \
                        drag_direction / np.linalg.norm(drag_direction)

                    left_end_drag_pos = left_start_drag_pos + drag_direction
                    right_end_drag_pos = right_start_drag_pos + drag_direction

                    final_drag_reachable =\
                        self.check_arm_reachability(
                            self.left_arm_base, left_end_drag_pos)\
                        and self.check_arm_reachability(
                            self.right_arm_base, right_end_drag_pos)
                    reachable = final_drag_reachable and reachable

                if not reachable:
                    print("REJECTED: unreachable")  #Test
                    continue
                action_kwargs['action_visualization'] =\
                    action_params['get_action_visualization_fn']()
                self.log_step_stats(action_kwargs)

                #an_tr_mo_4 Änderung
                # ============================================
                # Preserve pixel grasps for middleware/socket
                # ============================================

                pixels = action_params['pretransform_pixels'] #an_tr
                # ============================================
                # Remove internal-only keys before primitive execution
                # ============================================

                for k in ['valid_action',
                          'pretransform_pixels',
                          'get_action_visualization_fn']:
                    del action_params[k]
                # ============================================
                # Create middleware copy
                # ============================================

                middleware_action = action_params.copy()

                middleware_action['pretransform_pixels'] = pixels

                #Test-----------
                print("\nACCEPTED ACTION")
                #print("primitive:", action_kwargs['action_primitive'])
                print("MIDDLEWARE-pixels:", middleware_action['pretransform_pixels'])
                #print("\nRETVAL PRETRANSFORM PIXELS:", retval["pretransform_pixels"])
                #print("\nTRANSFORMED PIXELS:", pixels)
                #---------------
                print("\n===== VALID ACTION FOUND =====")       #Test Counter bis P2
                print("Candidates tested:", candidate_counter)
                print("Rejected candidates:", invalid_counter)
                print("BEST P1 RATIO SEEN:", self.best_p1_ratio)    #Test höhe counter
                print("BEST P2 RATIO SEEN:", self.best_p2_ratio)

                print("NETWORK P1:", p1)
                print("NETWORK P2:", p2)

                print("PRETRANSFORM:", middleware_action['pretransform_pixels'])
                print("P1_GRASP_CLOTH:", middleware_action['p1_grasp_cloth'])
                print("P2_GRASP_CLOTH:", middleware_action['p2_grasp_cloth'])
                print("\n===== ACTION TYPE =====")
                print("PRIMITIVE:", action_kwargs['action_primitive'])
                #print("PRIMITIVE CHANNEL:", indices[0])
                print("VALUE:", value.item())
                return action_kwargs['action_primitive'], middleware_action
                #OG teil:
                #return action_kwargs['action_primitive'], action_params
        return None, None

    def reset(self):
        self.episode_memory = Memory()
        self.episode_reward_sum = 0.
        self.current_timestep = 0
        self.terminate = False
        self.current_task = self.get_task_fn()
        if self.gui:
            print(self.current_task)
        set_scene(
            config=self.current_task.get_config(),
            state=self.current_task.get_state())

        self.init_coverage = get_current_covered_area(self.particle_radius)
        self.action_tool.reset([0.2, 0.5, 0.0])
        self.reset_end_effectors()

        self.step_simulation()
        self.set_grasp(False)
        self.env_video_frames = {}
        obs = self.get_obs()
        self.episode_memory.add_value(
            key='pretransform_observations', value=obs)
        self.pretransform_obs = obs.detach().cpu().numpy().copy() #für visualize
        print(self.pretransform_obs.shape) #Test
        self.transformed_obs = prepare_image(
            obs, self.get_transformations(), self.obs_dim,
            parallelize=self.parallelize_prepare_image)
        return self.transformed_obs, self.ray_handle

    def render_cloth(self):
        if self.render_engine == 'blender':
            mesh = get_cloth_mesh(*self.current_task.cloth_size)
            return blender_render_cloth(mesh, self.image_dim)
        elif self.render_engine == 'opengl':
            return get_image(self.image_dim, self.image_dim)
        else:
            raise NotImplementedError()

    def get_cloth_mask(self, rgb=None):
        if rgb is None:
            rgb = self.render_cloth()[0]
        bottom = (0, 0, 0)
        top = (100, 100, 100)
        mask = cv2.inRange(cv2.cvtColor(
            rgb, cv2.COLOR_RGB2HSV), bottom, top)
        mask = (mask == 0).astype(np.uint8)
        return get_largest_component(mask).astype(np.uint8)

    def get_obs(self):
        rgb, d = self.render_cloth()
        self.pretransform_rgb = rgb.copy() #Für Visualis
        self.pretransform_depth = d
        #self.pretransform_rgb = rgb #OG
        # cloths are closer than 2.0 meters from camera plane
        cloth_mask = self.get_cloth_mask(rgb)

        x, y = np.where(cloth_mask)
        dimx, dimy = self.pretransform_depth.shape

        self.adaptive_scale_factors = self.scale_factors.copy()
        if self.use_adaptive_scaling:
            try:
                # Minimum square crop
                cropx = max(dimx - 2*x.min(), dimx - 2*(dimx-x.max()))
                cropy = max(dimy - 2*y.min(), dimy - 2*(dimy-y.max()))
                crop = max(cropx, cropy)
                # Some breathing room
                crop = int(crop*1.5)
                if crop < dimx:
                    self.adaptive_scale_factors *= crop/dimx
                    #-----Test für scale----
                    print("\n===== ADAPTIVE SCALE DEBUG =====")
                    #print("crop:", crop)
                    #print("dimx:", dimx)
                    print("crop/dimx:", crop / dimx)
                    #print("scale_factors:", self.scale_factors)
                    #print("adaptive_scale_factors:", self.adaptive_scale_factors)
                    #-----TestEnde
                    self.episode_memory.add_value(
                        key='adaptive_scale',
                        value=float(crop/dimx))
            except Exception as e:
                print(e)
                print(self.current_task)

        return preprocess_obs(rgb.copy(), d.copy())

    def movep(self, pos, speed=None, limit=1000, min_steps=None, eps=1e-4):
        if speed is None:
            if self.dump_visualizations:
                speed = self.default_speed
            else:
                speed = 0.1
        target_pos = np.array(pos)
        for step in range(limit):
            curr_pos = self.action_tool._get_pos()[0]
            deltas = [(targ - curr)
                      for targ, curr in zip(target_pos, curr_pos)]
            dists = [np.linalg.norm(delta) for delta in deltas]
            if all([dist < eps for dist in dists]) and\
                    (min_steps is None or step > min_steps):
                return
            action = []
            for targ, curr, delta, dist, gs in zip(
                    target_pos, curr_pos, deltas, dists, self.grasp_states):
                if dist < speed:
                    action.extend([*targ, float(gs)])
                else:
                    delta = delta/dist
                    action.extend([*(curr+delta*speed), float(gs)])
            action = np.array(action)
            self.action_tool.step(action, step_sim_fn=self.step_simulation)
            if step % 4 == 0 and self.dump_visualizations:
                if 'top' not in self.env_video_frames:
                    self.env_video_frames['top'] = []
                self.env_video_frames['top'].append(
                    np.squeeze(np.array(get_image()[0])))
        raise MoveJointsException

    def reset_end_effectors(self):
        self.movep([[0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]], speed=5e-3)

    def set_grasp(self, grasp):
        if type(grasp) == bool:
            self.grasp_states = [grasp] * len(self.grasp_states)
        elif len(grasp) == len(self.grasp_states):
            self.grasp_states = grasp
        else:
            raise Exception()

    # def on_episode_end(self, log=False):
    #     if self.dump_visualizations and len(self.episode_memory) > 0:
    #         while True:
    #             hashstring = hashlib.sha1()
    #             hashstring.update(str(time()).encode('utf-8'))
    #             vis_dir = self.log_dir + '/' + hashstring.hexdigest()[:10]
    #             if not os.path.exists(vis_dir):
    #                 break
    #         os.mkdir(vis_dir)
    #         for key, frames in self.env_video_frames.items():
    #             if len(frames) == 0:
    #                 continue
    #             path = f'{vis_dir}/{key}.mp4'
    #             with imageio.get_writer(path, mode='I', fps=24) as writer:
    #                 for frame in (
    #                     frames if not log
    #                         else tqdm(frames, desc=f'Dumping {key} frames')):
    #                     writer.append_data(frame)
    #         self.episode_memory.add_value(
    #             key='visualization_dir',
    #             value=vis_dir)
    #     self.env_video_frames.clear()
    #     self.episode_memory.dump(
    #         self.replay_buffer_path)
    #     del self.episode_memory
    #     self.episode_memory = Memory()

    import cv2
    def on_episode_end(self, log=False):
        if self.dump_visualizations and len(self.episode_memory) > 0:
            while True:
                hashstring = hashlib.sha1()
                hashstring.update(str(time()).encode('utf-8'))
                vis_dir = os.path.join(self.log_dir, hashstring.hexdigest()[:10])
                if not os.path.exists(vis_dir):
                    break
            os.mkdir(vis_dir)

            for key, frames in self.env_video_frames.items():
                if len(frames) == 0:
                    continue

                # ============================================
                # TEST: save last video frame as REAL AFTER
                # NUTZEN: für Last Frame (nur für meine socket_eval)
                # ============================================

                last_frame = frames[-1]

                self.last_after_rgb = cv2.cvtColor(
                    last_frame,
                    cv2.COLOR_RGB2BGR
                )

                cv2.imwrite(
                    f"socket_eval/step_{self.current_step_id:03d}_REAL_AFTER.png",
                    self.last_after_rgb
                )
                print("REAL AFTER SAVED")
                #------TEST ENDE ------------
                path = os.path.join(vis_dir, f'{key}.mp4')
                height, width, _ = frames[0].shape

                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
                print("WRITING VIDEO:", path)           #Test prints Video
                print("FRAME SHAPE:", frames[0].shape)
                out = cv2.VideoWriter(path, fourcc, 24, (width, height))

                for frame in (frames if not log else tqdm(frames, desc=f'Dumping {key} frames')):
                    # Convert from RGB to BGR (OpenCV uses BGR)
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(bgr_frame)

                out.release()
                print("VIDEO SAVED:", path)     #Test Video

            self.episode_memory.add_value(
                key='visualization_dir',
                value=vis_dir)

        self.env_video_frames.clear()
        self.episode_memory.dump(self.replay_buffer_path)
        del self.episode_memory
        self.episode_memory = Memory()

    def is_cloth_grasped(self):
        positions = pyflex.get_positions().reshape((-1, 4))
        positions = positions[:, :3]
        heights = positions[:, 1]
        return heights.max() > 0.2

    def setup_ray(self, id):
        self.ray_handle = {"val": id}
