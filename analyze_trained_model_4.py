import shutil
import os
import random
import socket
import json
import torch
import numpy as np

from environment.tasks import TaskLoader
from environment.simEnv import SimEnv

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

    def reset(self, *args, **kwargs):

        obs, info = super().reset(*args, **kwargs)

        self.pretransform_obs = obs

        return obs, info

    def step(self, value_maps):

        # ============================================
        # RUN FLINGBOT STEP
        # ============================================
        obs, info = super().step(value_maps)

        #Test print
        print("CURRENT COVERAGE:", self.compute_coverage())
        print("RGB MEAN:", self.pretransform_rgb.mean())
        print("RGB STD :", self.pretransform_rgb.std())

        print("OBS MEAN:", self.transformed_obs.mean().item())
        print("OBS STD :", self.transformed_obs.std().item())

        self.pretransform_obs = obs

        # ============================================
        # SAVE AFTER FRAME
        # ============================================
        #-------------------------änderung von After damit es an Before passt!------------------
        import cv2

        step_id = self.current_step_id

        self.last_after_rgb = self.pretransform_rgb.copy()   #ersatz für After! return ist og

        print("AFTER RGB ID:", id(self.last_after_rgb))
        print("PRE RGB ID:", id(self.pretransform_rgb))
        print("AFTER RGB MEAN:", self.pretransform_rgb.mean())
        print("AFTER RGB STD :", self.pretransform_rgb.std())
        print("AFTER RGB SUM:", self.pretransform_rgb.sum())

        #img = self.pretransform_rgb.copy()
        img = self.last_after_rgb.copy()

        #img = (img * 255).clip(0,255).astype(np.uint8)    #zum testen auskommentiert, daraufhin waren alle bilder ident

        cv2.putText(
            img,
            f"STEP {step_id} AFTER",
            (20,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

        cv2.imwrite(
            f"socket_eval/step_{step_id:03d}_02_after.png",
            img
        )

        #self.last_after_rgb = self.pretransform_rgb.copy()   #ersatz für After! return ist og

        return obs, info

    def get_max_value_valid_action(self, value_maps):

        action_primitive, action = \
            super().get_max_value_valid_action(value_maps)

        print("PRETRANSFORM_RGB ID:", id(self.pretransform_rgb))
        print("OBS ID:", id(self.pretransform_obs))
        print("TRANSFORMED OBS SHAPE:", self.transformed_obs.shape)

        if action is not None:

            # ============================================
            # PIXEL GRASPS FROM FLINGBOT
            # ============================================

            pixels = action["pretransform_pixels"].copy()
            #pixels = np.array(pixels)   #Falls mal Listen statt Arrays kommen

            # ============================================
            # CLOTH VALIDITY CHECK
            # ============================================

            #mask = self.pretransform_rgb.mean(axis=2)

            #THRESHOLD = 40

            #p1x, p1y = pixels[0].astype(int)
            #p2x, p2y = pixels[1].astype(int)

            # Sicherheitsclamp
            #H, W = mask.shape

            #p1x = np.clip(p1x, 0, W-1)
            #p1y = np.clip(p1y, 0, H-1)

            #p2x = np.clip(p2x, 0, W-1)
            #p2y = np.clip(p2y, 0, H-1)

            #p1_valid = mask[p1y, p1x] > THRESHOLD
            #p2_valid = mask[p2y, p2x] > THRESHOLD

            #print("\n=== CLOTH CHECK ===")
            #print("P1 VALID:", p1_valid)
            #print("P2 VALID:", p2_valid)

            #if not (p1_valid and p2_valid):
            #    print("REJECTING: POINT OUTSIDE CLOTH")
            #    #return None, None


            # Test visual für echte replayframes, png speicherung, orientierung und generell bild
            import cv2

            step_id = len(self.grasp_log)
            self.current_step_id = step_id
            # bereits NumPy array mit shape: (4,400,400)
           # img = self.pretransform_obs                              #vor visual
           # img = np.swapaxes(img, 0, -1)
           # img = (img[:, :, :3] * 255).astype(np.uint8)

            #----------------------- damit after.img mein before(n+1) wird-------------------
            # Für Step 0 existiert noch kein After-Bild
            if hasattr(self, "last_after_rgb"):
                img = self.last_after_rgb.copy()

                print("BEFORE RGB ID:", id(img))
                print("LAST AFTER ID:", id(self.last_after_rgb))

            else:
                img = self.pretransform_rgb.copy()

            #img = self.pretransform_rgb.copy()   # alter code nur before
            print("RAW_MAX:", img.max())
            print("RAW_MIN:", img.min())
            print("RAW_DTYPE:", img.dtype)
            print("IMG SHAPE:", img.shape)

            # ============================================
            # DRAW GRASP POINTS
            # ============================================

            p1_int = tuple(pixels[0][::-1].astype(int))
            #p1_int = tuple(pixels[0].astype(int)) #Test für Achsen
            #print("p1_int:", p1_int) #Same wie float und payload
            p2_int = tuple(pixels[1][::-1].astype(int))
            #p2_int = tuple(pixels[1].astype(int))
            #print("p2_int", p2_int)

            # p1 = green         #noch keine Greifer L R zuweisung!!
            cv2.circle(
                img,
                p1_int,
                radius=6,
                color=(0,255,0),
                thickness=-1
            )

            # p2 = red            #nicht rot! blau
            cv2.circle(
                img,
                p2_int,
                radius=6,
                color=(255,0,0),
                thickness=-1
            )

            # line between grasp points
            cv2.line(
                img,
                p1_int,
                p2_int,
                color=(255,255,0),
                thickness=2
            )

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

            cv2.imwrite(
                f"socket_eval/step_{step_id:03d}_01_before.png",
                img
            )

            # ============================================
            # GRASP PIXELS
            # ============================================
            p1 = pixels[0].astype(np.float32)
            #print("p1_float:", p1) #Same wie int und payload
            p2 = pixels[1].astype(np.float32)
            #print("p2_float:", p2)

            # ============================================
            # FINAL ROBOT CAMERA PIXELS
            # ============================================
#            p1_robot = p1.astype(int)
#            p2_robot = p2.astype(int)

            #---------------------------------------------
            # RENDER_DIM ADJUSTMENT
            #---------------------------------------------
            CAMERA_TARGET = 720
            RENDER_DIM = self.image_dim          #um Hardcoden auf 400 zu vermeiden

            camera_scale = CAMERA_TARGET / RENDER_DIM

            p1_robot = (p1 * camera_scale).astype(int)
            p2_robot = (p2 * camera_scale).astype(int)
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

                "primitive": action_primitive,

                "p1_px": p1_robot.tolist(),

                "p2_px": p2_robot.tolist()
            }

            print("\n========== RAW ACTION DEBUG ==========")

            print("ACTION KEYS:", action.keys())

            print("\n========== WORLD SPACE ==========") #= PIXEL_TO_3D Raw (erwählte points)
            print("WORLD P1:", action["p1"])
            print("WORLD P2:", action["p2"])

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

            print("\n[SOCKET SENT]", payload)

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

    gui=args.gui #False
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
    diff = torch.mean(torch.abs(
        env.transformed_obs - prev_obs
    ))

    print("OBS DIFF:", diff.item())

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
