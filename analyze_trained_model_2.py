# ##Trajektorien_ unnötig als log_ aber momentan lassen

import json
import torch
import pyflex

from environment.tasks import TaskLoader
from environment.simEnv import SimEnv

from utils import config_parser, setup_network


# ============================================================
# 1. ARGUMENTE / TRAINIERTES MODELL
# ============================================================

args = config_parser().parse_args()


# ============================================================
# 2. EIN KLEIDUNGSSTÜCK LADEN
# ============================================================

loader = TaskLoader(
    "flingbot-normal-rect-eval.hdf5",
    repeat=False
)

task = loader.get_next_task()


# ============================================================
# 3. DEBUG / ANALYSE ENVIRONMENT
# ============================================================

class DebugSimEnv(SimEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Finale Greifpunkte
        self.grasp_log = []

        # Komplette Trajektorien
        self.motion_log = []

    # --------------------------------------------------------
    # Finale intelligente Greifpunkte loggen
    # --------------------------------------------------------

    def get_max_value_valid_action(self, value_maps):

        action_primitive, action = \
            super().get_max_value_valid_action(value_maps)

        if action is not None:

            print("\n========== TRAINED GRASP ==========")

            print("LEFT  3D:", action["p1"])
            print("RIGHT 3D:", action["p2"])

            self.grasp_log.append({

                "primitive": action_primitive,

                "left_grasp": action["p1"].tolist(),

                "right_grasp": action["p2"].tolist()
            })

        return action_primitive, action

    # --------------------------------------------------------
    # KOMPLETTE GREIFERTRAJEKTORIEN
    # --------------------------------------------------------

    def movep(self, pos, *args, **kwargs):

        # Aktuelle Greiferpositionen
        curr_pos = self.action_tool._get_pos()[0]

        # Cloth Partikelzustände
        cloth_positions = pyflex.get_positions().reshape(-1, 4)

        self.motion_log.append({

            # Zeitpunkt
            "step": len(self.motion_log),

            # Linker Greifer
            "left_gripper": curr_pos[0].tolist(),

            # Rechter Greifer
            "right_gripper": curr_pos[1].tolist(),

            # Kompletter Clothzustand
            "cloth_particles": cloth_positions.tolist()
        })

        return super().movep(pos, *args, **kwargs)


# ============================================================
# 4. SIMULATION ENVIRONMENT
# ============================================================

env = DebugSimEnv(

    replay_buffer_path="outputs/debug_replay.hdf5",

    obs_dim=128,

    num_rotations=16,

    scale_factors=[1.0],

    # IMMER DAS GLEICHE CLOTH
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
# 5. TRAINIERTES MODELL ÜBER setup_network()
# ============================================================

policy, optimizer, dataset_path = setup_network(args)

policy.eval()

print("\nTrainiertes Modell geladen.")


# ============================================================
# 6. CLOTH RESETTEN
# ============================================================

obs, _ = env.reset()

print("\nCloth geladen.")
print(task)


# ============================================================
# 7. VALUE MAPS VOM TRAINIERTEN MODELL
# ============================================================

with torch.no_grad():

    value_maps = policy.act([env.transformed_obs])[0]


# ============================================================
# 8. KOMPLETTE FLING AKTION AUSFÜHREN
# ============================================================

env.step(value_maps)


# ============================================================
# 9. FINALE GREIFPUNKTE SPEICHERN
# ============================================================

with open("outputs/trained_grasps.json", "w") as f:

    json.dump(env.grasp_log, f, indent=4)


# ============================================================
# 10. KOMPLETTE TRAJEKTORIEN SPEICHERN
# ============================================================

with open("outputs/motion_log.json", "w") as f:

    json.dump(env.motion_log, f, indent=4)


print("\n======================================")
print("Analyse abgeschlossen.")
print("Gespeichert:")
print(" - outputs/trained_grasps.json")
print(" - outputs/motion_log.json")
print("======================================")
