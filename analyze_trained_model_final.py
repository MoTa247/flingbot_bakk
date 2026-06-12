import shutil
import os
import random
import socket
import json
import torch
import numpy as np

from environment.tasks import TaskLoader
from environment.simEnv import SimEnv
from environment.utils import draw_action

from utils import config_parser, setup_network

args = config_parser().parse_args()

# ================================
# Neueste Task Load überschreibt die Daten des letzen Loads
# ================================

OUTPUT_DIR = "socket_eval"

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR)



# ============================================================
# 1. TASK LOAD
# ============================================================

AVAILABLE_TASKS = [
    "normal-rect",
    "large-rect",
    "shirt"
]

TASK_NAME = random.choice(AVAILABLE_TASKS)
print("DEBUG TASK:", TASK_NAME)
print(f"\nGewählte Task-Kategorie: {TASK_NAME}")

TASK_FILE = f"flingbot-{TASK_NAME}-eval.hdf5"

loader = TaskLoader(
    TASK_FILE,
    repeat=False
)

task = loader.get_next_task()

# ============================================================
# 2. SOCKET CONNECT
# ============================================================

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

client_socket.connect(("172.17.0.1", 5001))

print("\n[SOCKET] Connected to Robot Controller")

# ============================================================
# 3. DEBUG ENVIRONMENT
# ============================================================

class DebugSimEnv(SimEnv):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.grasp_log = []

    def dump_current_video(self, step_id):

        if 'top' not in self.env_video_frames:
            print("NO VIDEO FRAMES")
            return

        frames = self.env_video_frames['top']

        if len(frames) == 0:
            print("EMPTY VIDEO FRAMES")
            return

        path = f"socket_eval/step_{step_id:03d}.mp4"

        height, width, _ = frames[0].shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            path,
            fourcc,
            24,
            (width, height)
        )

        for frame in frames:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr)

        out.release()

        print("VIDEO SAVED:", path)

    def reset(self, *args, **kwargs):

        obs, info = super().reset(*args, **kwargs)

        self.pretransform_obs = obs

	# semantisch sauber
        self.before_rgb = self.pretransform_rgb.copy()

        return obs, info
        
    def step(self, value_maps):

        # ============================================
        # RUN FLINGBOT STEP
        # ============================================
        obs, info = super().step(value_maps)

        self.pretransform_obs = obs

        # ============================================
        # SAVE AFTER FRAME
        # ============================================
        import cv2

        step_id = self.current_step_id

        if hasattr(self, "last_after_rgb"):
            img = self.last_after_rgb.copy()
        else:
            img = self.pretransform_rgb.copy()

        cv2.putText(
            img,
            f"STEP {step_id} AFTER",
            (20,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

        print("\n===== SAVE DEBUG =====")
        print("STEP:", step_id)
        print("SAVE TYPE:", "AFTER")
        print("IMAGE ID:", id(img))
        print("IMAGE SHAPE:", img.shape)
        print("IMAGE SUM:", img.sum())

        #-------------IMG rotation-------
        img_rot = cv2.rotate(
            img,
            cv2.ROTATE_90_CLOCKWISE
        )

        cv2.imwrite(
            f"socket_eval/step_{step_id:03d}_02_after.png",
            img_rot #img
        )

        # ============================================
        # SAVE ACTION VIDEO
        # ============================================

        if 'top' in self.env_video_frames:

            frames = self.env_video_frames['top']
            if len(frames) > 0:
                 video_path = f"socket_eval/step_{step_id:03d}.mp4"
                 height, width, _ = frames[0].shape
                 fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                 out = cv2.VideoWriter(
                     video_path,
                     fourcc,
                     24,
                     (width, height)
                 )
                 for frame in frames:
                     bgr_frame = cv2.cvtColor(
                         frame,
                         cv2.COLOR_RGB2BGR
                     )
                     out.write(bgr_frame)

                 out.release()

        self.env_video_frames.clear()

        return obs, info

    def get_max_value_valid_action(self, value_maps):

        action_primitive, action = \
            super().get_max_value_valid_action(value_maps)

        if action is not None:

            # ============================================
            # PIXEL GRASPS FROM FLINGBOT
            # ============================================

            pixels = action["pretransform_pixels"].copy()
            
            # echtes OG-Bild sichern
            self.before_rgb = self.pretransform_rgb.copy()

            # ============================================
            # CLOTH VALIDITY CHECK
            # ============================================

            # Test visual für echte replayframes, png speicherung, orientierung und generell bild
            import cv2

            step_id = len(self.grasp_log)
            self.current_step_id = step_id

            if hasattr(self, "last_after_rgb"):
                img = self.last_after_rgb.copy()
            else:
                img = self.pretransform_rgb.copy()
	    
            img = self.before_rgb.copy()

            cloth_y, cloth_x = np.where(self.pretransform_depth < 1.99)

            if len(cloth_y) > 0:
                print(
                    "CLOTH BBOX:",
                    "y=[", cloth_y.min(), ",", cloth_y.max(), "]",
                    "x=[", cloth_x.min(), ",", cloth_x.max(), "]"
                )

	    # ============================================
	    # DRAW ACTION (OG FLINGBOT)
	    # ============================================

            overlay = draw_action(
               action_primitive=action_primitive,
               shape=self.pretransform_depth.shape[:2],
               pixels=pixels,
               thickness=3
            )
	    
            overlay_rgb = (overlay[:, :, :3] * 255).astype(np.uint8)
            mask = overlay_rgb.sum(axis=2) > 0

            img[mask] = overlay_rgb[mask]
	    
            # ============================================
            # DRAW LABEL
            # ============================================

            # step label
            cv2.putText(
                img,
                f"STEP {step_id}",
                (20,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255),
                2
            )

            print("\n===== SAVE DEBUG =====")
            print("STEP:", step_id)
            print("SAVE TYPE:", "BEFORE")
            print("IMAGE ID:", id(img))
            print("IMAGE SHAPE:", img.shape)
            print("IMAGE SUM:", img.sum())

            print("BEFORE FILE:", f"socket_eval/step_{step_id:03d}_01_before.png") #Test Before 00 finden
            #---------IMG ROTATION----------
            img_rot = cv2.rotate(
                img,
                cv2.ROTATE_90_CLOCKWISE
            )

            cv2.imwrite(
                f"socket_eval/step_{step_id:03d}_01_before.png",
                img_rot #img
            )

            # ============================================
            # GRASP PIXELS
            # ============================================
            p1 = pixels[0].astype(np.float32)
            p2 = pixels[1].astype(np.float32)

            #TEST Rotation bei übertragung erkennen
            print("P1:", p1)
            print("P2:", p2)

            vec = p2 - p1

            print("VECTOR:", vec)

            angle = np.degrees(
                np.arctan2(
                    vec[1],
                    vec[0]
                )
            )

            print("ANGLE:", angle)
            #ENDE TEST ROTATION

	    #---------------------------------------------
            # RENDER_DIM ADJUSTMENT
            #---------------------------------------------
            CAMERA_TARGET = 720
            RENDER_DIM = self.image_dim          #um Hardcoden auf 400 zu vermeiden

            camera_scale = CAMERA_TARGET / RENDER_DIM

            p1_robot = (p1 * camera_scale).astype(int)
            p2_robot = (p2 * camera_scale).astype(int)

            # ---------------------------------------------
            # KOORDINATENANPASSUNG: 
            # ---------------------------------------------
            p1_robot = np.array([ p1[1], p1[0] ]) 
            p2_robot = np.array([ p2[1], p2[0] ])
            #p1_robot = np.array([
            #    p1_robot[1],
            #    CAMERA_TARGET - 1 - p1_robot[0]
            #])

            #p2_robot = np.array([
            #    p2_robot[1],
            #    CAMERA_TARGET - 1 - p2_robot[0]
            #])

            p1_robot = (p1_robot * camera_scale).astype(int)
            p2_robot = (p2_robot * camera_scale).astype(int)

            #TEST ÜBERTRAGUNG ROTATION
            print("\n============================")
            print("ROBOT PIXELS")
            print("============================")

            print("P1_ROBOT:", p1_robot)
            print("P2_ROBOT:", p2_robot)

            #vec = p2_robot - p1_robot

            #print("VECTOR:", vec)

            #angle = np.degrees(
            #    np.arctan2(
            #        vec[1],
            #        vec[0]
            #    )
            #)

            #print("ANGLE:", angle)
            #ENDE TEST ROTATION
            #---------------------------------------------
            # RENDER_DIM -> ROBOT CAMERA SPACE             (bis vor payload)
            # ---------------------------------------------

#            CAMERA_WIDTH  = 1280
#            CAMERA_HEIGHT = 720

#            RENDER_DIM = self.image_dim

            # Scale über Höhe
#            camera_scale = CAMERA_HEIGHT / RENDER_DIM

            # Resultierende quadratische Bildbreite
#            scaled_render_width = RENDER_DIM * camera_scale

            # Horizontales Letterbox-Padding
#            padding_x = int(
#                (CAMERA_WIDTH - scaled_render_width) / 2
#            )

            # ---------------------------------------------
            # FINAL CAMERA PIXELS
            # ---------------------------------------------

#            p1_robot = np.array([
#                int(p1[0] * camera_scale + padding_x),
#                int(p1[1] * camera_scale)
#            ])

#            p2_robot = np.array([
#                int(p2[0] * camera_scale + padding_x),
#                int(p2[1] * camera_scale)
#            ])

            payload = {
                "action_id": len(self.grasp_log),

                "primitive": action_primitive,

                "p1_px": p1_robot.tolist(),

                "p2_px": p2_robot.tolist()
            }

            #print("\n========== RAW ACTION DEBUG ==========")

            #print("ACTION KEYS:", action.keys())

            #print("\n========== WORLD SPACE ==========") #= PIXEL_TO_3D Raw (erwählte points)
            #print("WORLD P1:", action["p1"])
            #print("WORLD P2:", action["p2"])

            #print("\nPRETRANSFORM PIXELS:")    #Ident zu pixel aus payload "p1_px" und "p2_px" mit p1_robot.tolist()
            print(action["pretransform_pixels"])

#            print("\n========== TRAINED ACTION ==========")   #ident Pretransform Pixels
#
#            print("Primitive :", action_primitive)

#            print("LEFT PIXEL :", pixels[0])
#            print("RIGHT PIXEL:", pixels[1])

#            print("\n========== RENDER SPACE ==========") #ident Pretransform Pixels
#            print("POLICY P1:", pixels[0])
#            print("POLICY P2:", pixels[1])

            # ============================================
            # SOCKET PAYLOAD
            # ============================================
            # PAYLOAD BENEATH ROBOT CAMERA SPACE
            #-------------------------------------

            client_socket.send(
                (json.dumps(payload) + "\n").encode()
            )
            while True:
               ack_data = client_socket.recv(4096).decode().strip()
               if not ack_data:
                  raise RuntimeError("Robot socket closed while waiting for ACK")
               ack = json.loads(ack_data)
               print("\n[ACK RECEIVED]", ack)
               if ack.get("status") == "executed":
                  break
            #print("\n[SOCKET SENT]", payload)

            # ============================================
            # LOGGING
            # ============================================

            self.grasp_log.append({

                "primitive": action_primitive,

                "policy_p1_px": pixels[0].tolist(),
                "policy_p2_px": pixels[1].tolist(),

                "robot_p1_px": p1_robot.tolist(),
                "robot_p2_px": p2_robot.tolist()
            })
            # Remove middleware-only key before primitive execution
            del action["pretransform_pixels"]

        return action_primitive, action


# ============================================================
# 4. ENVIRONMENT
# ============================================================            

env = DebugSimEnv(

    replay_buffer_path="outputs/debug_replay.hdf5",

    obs_dim=args.obs_dim, #64, #128,

    num_rotations=args.num_rotations, #12, #16,

    scale_factors=args.scale_factors, #[1.0,1.25,1.50,1.75,2.00,2.25,2.50,2.75],

    get_task_fn=lambda: task,

    action_primitives=[
        'fling',
        'drag',
        'stretchdrag',
        'place'
    ],

    pix_grasp_dist=args.pix_grasp_dist, #16,

    pix_drag_dist=args.pix_drag_dist, #16,

    pix_place_dist=args.pix_place_dist, #10,

    stretchdrag_dist=args.stretchdrag_dist, #0.3,

    reach_distance_limit=args.reach_distance_limit, #1.0,

    fixed_fling_height=args.fixed_fling_height, #0.7,

    render_engine='opengl',

    gui=args.gui, #False,

    dump_visualizations=True      # <-- neu
)

# ============================================================
# 5. LOAD NETWORK
# ============================================================            

policy, optimizer, dataset_path = setup_network(args)

policy.eval()

print("\nTrainiertes Modell geladen.")

# ============================================================
# 6. RESET CLOTH
# ============================================================

obs, _ = env.reset()

# ============================================================
# 7. INITIAL FORWARD PASS
# ============================================================            

with torch.no_grad():

    transformed_obs = env.transformed_obs.cuda()

    prev_obs = env.transformed_obs.clone() #Test policy
    value_maps = policy.act([env.transformed_obs])[0]
    #Test policy bis print
    #diff = torch.mean(torch.abs(
    #    env.transformed_obs - prev_obs
    #))

    #print("OBS DIFF:", diff.item())

# ============================================================
# 8. MULTI ACTION LOOP
# ============================================================    

MAX_ACTIONS = 10

for action_step in range(MAX_ACTIONS):

    print("\nSTARTING NEXT ACTION")

    print(f"\n========== ACTION {action_step} ==========")

    # ========================================================
    # NEW VALUE MAPS
    # ========================================================

    with torch.no_grad():

        value_maps = policy.act([env.transformed_obs])[0]


    # ========================================================
    # EXECUTE ACTION
    # ========================================================

    prev_obs = env.transformed_obs.clone() #Test policy
    obs, info = env.step(value_maps)
    #Test policy bis print
    diff = torch.mean(torch.abs(
        env.transformed_obs - prev_obs
    ))

    print("\nOBS DIFF:", diff.item())

    # ========================================================
    # COVERAGE
    # ========================================================

    try:

        coverage = info['normalised_coverage']

        print(f"Coverage: {coverage:.4f}")

    except:

        pass

# ============================================================
# 9. SAVE LOG
# ============================================================

with open("outputs/trained_grasps.json", "w") as f:

    json.dump(env.grasp_log, f, indent=4)

print("\nGreifpunkte gespeichert.")

# ============================================================
# 10. END SIGNAL
# ============================================================

client_socket.send(

    (json.dumps({
        "done": True
    }) + "\n").encode()

)

print("\n[SOCKET] End signal sent.")

client_socket.close()

print("[SOCKET] Closed.")
