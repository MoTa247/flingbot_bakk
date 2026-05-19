import random
import socket
import json
import torch
import numpy as np

from environment.tasks import TaskLoader
from environment.simEnv import SimEnv

from utils import config_parser, setup_network


# ============================================================
# 1. ARGUMENTE
# ============================================================

args = config_parser().parse_args()


# ============================================================
# 2. TASK AUSWÄHLEN
# ============================================================

AVAILABLE_TASKS = [
    "normal-rect",
    "large-rect",
    "shirt"
]

TASK_NAME = random.choice(AVAILABLE_TASKS)

print(f"\nGewählte Task-Kategorie: {TASK_NAME}")

TASK_FILE = f"flingbot-{TASK_NAME}-eval.hdf5"

loader = TaskLoader(
    TASK_FILE,
    repeat=False
)

task = loader.get_next_task()


# ============================================================
# 3. SOCKET CLIENT
# ============================================================

client_socket = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM
)

client_socket.connect(("172.17.0.1", 5001))

print("\n[SOCKET] Connected to Robot Controller")


# ============================================================
# 4. DEBUG ENVIRONMENT
# ============================================================

class DebugSimEnv(SimEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.grasp_log = []

    # --------------------------------------------------------
    # Greifentscheidung abfangen
    # --------------------------------------------------------

    def get_max_value_valid_action(self, value_maps):

        stacked_value_maps = torch.stack(tuple(value_maps.values()))

        # --------------------------------------------------------
        # Randpixel entfernen
        # --------------------------------------------------------

        stacked_value_maps = stacked_value_maps[
            :,
            :,
            self.pix_grasp_dist:-self.pix_grasp_dist,
            self.pix_grasp_dist:-self.pix_grasp_dist
        ]

        sorted_values, _ = stacked_value_maps.flatten().sort(
            descending=True
        )

        actions = list(value_maps.keys())

        # --------------------------------------------------------
        # Beste Aktion suchen
        # --------------------------------------------------------

        for value in sorted_values:

            for indices in np.array(
                np.where(stacked_value_maps == value)
            ).T:

                # Randkorrektur
                indices[-2:] += self.pix_grasp_dist

                max_indices = indices[1:]

                x, y, z = max_indices

                action = actions[indices[0]]

                # ------------------------------------------------
                # PIXEL GREIFPUNKTE
                # ------------------------------------------------

                reach_points = np.array(

                    self.get_action_params(
                        action_primitive=action,
                        max_indices=(x, y, z)
                    )

                )

                p1_pix, p2_pix = reach_points[:2]

                print("\n========== PIXEL GRASP ==========")

                print("P1 PIX:", p1_pix)

                print("P2 PIX:", p2_pix)

                # ------------------------------------------------
                # CENTER / WINKEL / BREITE
                # ------------------------------------------------
                scale = 512/128

                center_px = [

                    int(((p1_pix[0] + p2_pix[0]) / 2)* scale),

                    int(((p1_pix[1] + p2_pix[1]) / 2)* scale)
                ]

                dx = p2_pix[0] - p1_pix[0]

                dy = p2_pix[1] - p1_pix[1]

                theta = float(np.arctan2(dy, dx))

                width_px = float(

                    np.linalg.norm(

                        np.array(p2_pix) - np.array(p1_pix)

                    )
                )
                width_px = width_px * scale    #128x128 auf 720x720 für MuJoCo

                # ------------------------------------------------
                # SOCKET SEND
                # ------------------------------------------------
                # TESTPIXEL
#                center_px = [360, 360]

                payload = {

                    "center_px": center_px,

                    "theta": theta,

                    "width_px": width_px
                }

                client_socket.send(

                    (json.dumps(payload)+"\n").encode()

                )

                print("\n[SOCKET SENT]")

                print(payload)

                print("AFTER SOCKET SEND")

                # ------------------------------------------------
                # LOGGING
                # ------------------------------------------------

                self.grasp_log.append({

                    "primitive": action,

                    "center_px": center_px,

                    "theta": theta,

                    "width_px": width_px
                })

                # ------------------------------------------------
                # WICHTIG:
                # Echte Aktion weiter normal ausführen
                # ------------------------------------------------

                return super().get_max_value_valid_action(
                    value_maps
                )

        return None, None

# ============================================================
# 5. ENVIRONMENT
# ============================================================

env = DebugSimEnv(

    replay_buffer_path="outputs/debug_replay.hdf5",

    obs_dim=128,

    num_rotations=16,

    scale_factors=[1.0],

    get_task_fn=lambda: task,

    action_primitives=['fling'],

    pix_grasp_dist=16,

    pix_drag_dist=16,

    pix_place_dist=10,

    stretchdrag_dist=0.3,

    reach_distance_limit=1.0,

    fixed_fling_height=0.7,

    render_engine='opengl',

    gui=False
)


# ============================================================
# 6. MODELL LADEN
# ============================================================

policy, optimizer, dataset_path = setup_network(args)

policy.eval()

print("\nTrainiertes Modell geladen.")


# ============================================================
# 7. ENV RESET
# ============================================================

obs, _ = env.reset()


# ============================================================
# 8. MEHRERE GREIFAKTIONEN
# ============================================================

MAX_ACTIONS = 10

for action_step in range(MAX_ACTIONS):

    print("STARTING NEXT ACTION")

    print(f"\n========== ACTION {action_step} ==========")

    with torch.no_grad():

        value_maps = policy.act([env.transformed_obs])[0]

    obs, info = env.step(value_maps)

    try:

        coverage = info['normalised_coverage']

        print(f"Coverage: {coverage:.4f}")

    except:
        pass


# ============================================================
# 9. LOG SPEICHERN
# ============================================================

with open("outputs/trained_grasps.json", "w") as f:

    json.dump(env.grasp_log, f, indent=4)

print("\nGreifpunkte gespeichert.")
