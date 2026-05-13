import random
import socket
import json
import torch
import numpy as np

from environment.tasks import TaskLoader
from environment.simEnv import SimEnv

from utils import config_parser, setup_network

args = config_parser().parse_args()

# -------------------------------------------------
# 1. Task laden
# -------------------------------------------------

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

# ----------------------------
# 2.1 Platz für Socket
# ----------------------------

#client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#client_socket.connect(("127.0.0.1", 5000))



# -------------------------------------------------
# 2. Sim Environment
# -------------------------------------------------

class DebugSimEnv(SimEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.grasp_log = []

    def get_max_value_valid_action(self, value_maps):

        action_primitive, action = \
            super().get_max_value_valid_action(value_maps)

        if action is not None:

            print("\n========== TRAINED GRASP ==========")

            print("LEFT :", action["p1"])
            print("RIGHT:", action["p2"])

#----------------------------------------------
#	client_socket.send(
#
#    	json.dumps({
#
#        	"left_grasp": action["p1"].tolist(),
#
#	        "right_grasp": action["p2"].tolist()
#
#   	 }).encode()
#
#	)
#----------------------------------------------




            self.grasp_log.append({
                "primitive": action_primitive,
                "left_grasp": action["p1"].tolist(),
                "right_grasp": action["p2"].tolist()
            })

        return action_primitive, action


# -------------------------------------------------
# 3. Environment erzeugen
# -------------------------------------------------

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


# -------------------------------------------------
# 4. Trainiertes Modell laden
# -------------------------------------------------

policy, optimizer, dataset_path = setup_network(args)

policy.eval()

print("\nTrainiertes Modell geladen.")


# -------------------------------------------------
# 5. Cloth resetten
# -------------------------------------------------

obs, _ = env.reset()


# -------------------------------------------------
# 6. Netzwerk Vorhersage
# -------------------------------------------------

with torch.no_grad():

    transformed_obs = env.transformed_obs.cuda()

    value_maps = policy.act([env.transformed_obs])[0]


# ============================================================
# 7. MEHRERE INTELLIGENTE GREIFAKTIONEN
# ============================================================

MAX_ACTIONS = 10

for action_step in range(MAX_ACTIONS):

    print(f"\n========== ACTION {action_step} ==========")

    # --------------------------------------------------------
    # Neue Value Maps vom aktuellen Cloth Zustand
    # --------------------------------------------------------

    with torch.no_grad():

        value_maps = policy.act([env.transformed_obs])[0]

    # --------------------------------------------------------
    # Aktion ausführen
    # --------------------------------------------------------

    obs, info = env.step(value_maps)

    # --------------------------------------------------------
    # Coverage ausgeben
    # --------------------------------------------------------

    try:
        coverage = info['normalised_coverage']

        print(f"Coverage: {coverage:.4f}")

    except:
        pass

    # --------------------------------------------------------
    # Abbruch wenn Episode fertig
    # --------------------------------------------------------

    #if done:

     #   print("\nEpisode beendet.")
      #  break

# -------------------------------------------------
# 8. Speichern
# -------------------------------------------------

with open("outputs/trained_grasps.json", "w") as f:
    json.dump(env.grasp_log, f, indent=4)

print("\nGreifpunkte gespeichert.")
