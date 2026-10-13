"""Count how often two agents' most confident predictions land on top of each other.

Usage:
    python measure_collisions.py <predictions.pkl> <gt.pkl> [threshold_m] [cases.csv]

    threshold_m  distance in metres that counts as a "collision" (default 2.5)
    cases.csv    optional: save every predicted vehicle-vehicle collision to this file,
                 so you can look at individual scenes

Runs the same check on the real future positions (from gt.pkl) for comparison.
"""
import csv
import pickle
import sys
from itertools import combinations

import numpy as np

from utils import unpack_npy

# Positions of the checkpoints inside the 16-step (0.5 s ... 8 s) trajectories
CHECKPOINTS = {'3s': 5, '5s': 9, '8s': 15}
TYPE_NAMES = {1: 'vehicle', 2: 'pedestrian', 3: 'cyclist'}


def pair_type(type_a, type_b):
    names = sorted([TYPE_NAMES.get(type_a, 'other'), TYPE_NAMES.get(type_b, 'other')])
    return '-'.join(names)


def main(pred_path, gt_path, threshold_m=2.5, cases_path=None):
    with open(pred_path, 'rb') as f:
        pred = pickle.load(f)
    with open(gt_path, 'rb') as f:
        gt = pickle.load(f)

    scenes = 0
    scenes_pred_hit = 0
    scenes_real_hit = 0
    pairs_by_type = {}   # pair type -> [pairs checked, predicted hits, real hits]
    vv_by_checkpoint = {name: [0, 0] for name in CHECKPOINTS}  # vehicle-vehicle [pred, real]
    cases = []

    for scene_id, agents in pred.items():
        scenes += 1
        top, conf, real, kind = {}, {}, {}, {}

        for agent_id, p in agents.items():
            pos = unpack_npy(p['pos'])            # [6, 16, 2]: 6 trajectories
            pi = unpack_npy(p['pi']).reshape(-1)  # [6]: their confidences
            best = int(np.argmax(pi))
            top[agent_id] = pos[best]             # the most confident trajectory
            conf[agent_id] = float(pi[best])

            g = gt[scene_id][agent_id]
            real[agent_id] = (unpack_npy(g['eval_pos']).reshape(-1, 2),
                              unpack_npy(g['eval_valid']).reshape(-1))
            kind[agent_id] = int(unpack_npy(g['agent_type']))

        pred_hit_in_scene = real_hit_in_scene = False

        for a, b in combinations(top, 2):
            key = pair_type(kind[a], kind[b])
            counts = pairs_by_type.setdefault(key, [0, 0, 0])
            counts[0] += 1
            (pos_a, ok_a), (pos_b, ok_b) = real[a], real[b]

            pred_hit = real_hit = False
            for name, i in CHECKPOINTS.items():
                d_pred = float(np.linalg.norm(top[a][i] - top[b][i]))
                p_hit = d_pred < threshold_m
                r_hit = bool(ok_a[i] and ok_b[i]
                             and np.linalg.norm(pos_a[i] - pos_b[i]) < threshold_m)
                if key == 'vehicle-vehicle':
                    vv_by_checkpoint[name][0] += p_hit
                    vv_by_checkpoint[name][1] += r_hit
                    if p_hit:
                        d_real = (float(np.linalg.norm(pos_a[i] - pos_b[i]))
                                  if ok_a[i] and ok_b[i] else float('nan'))
                        cases.append([scene_id, a, b, name, round(d_pred, 2),
                                      round(d_real, 2), round(conf[a], 3), round(conf[b], 3)])
                pred_hit |= p_hit
                real_hit |= r_hit

            counts[1] += pred_hit
            counts[2] += real_hit
            pred_hit_in_scene |= pred_hit
            real_hit_in_scene |= real_hit

        scenes_pred_hit += pred_hit_in_scene
        scenes_real_hit += real_hit_in_scene

    print(f'Threshold: {threshold_m} m')
    print(f'Scenes checked: {scenes}')
    print(f'Scenes with a predicted collision: {scenes_pred_hit} ({100 * scenes_pred_hit / scenes:.1f}%)')
    print(f'Scenes where agents were really that close: {scenes_real_hit} ({100 * scenes_real_hit / scenes:.1f}%)')
    print()
    print(f'{"pair type":<24}{"pairs":>8}{"predicted":>12}{"real":>8}')
    for key, (n, p, r) in sorted(pairs_by_type.items()):
        print(f'{key:<24}{n:>8}{p:>12}{r:>8}')
    print()
    print('vehicle-vehicle, by checkpoint (pairs too close at that moment):')
    print(f'{"checkpoint":<12}{"predicted":>12}{"real":>8}')
    for name, (p, r) in vv_by_checkpoint.items():
        print(f'{name:<12}{p:>12}{r:>8}')

    if cases_path:
        with open(cases_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['scenario_id', 'agent_a', 'agent_b', 'checkpoint',
                        'predicted_distance_m', 'real_distance_m', 'conf_a', 'conf_b'])
            w.writerows(cases)
        print(f'\nSaved {len(cases)} vehicle-vehicle cases to {cases_path}')


if __name__ == '__main__':
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 2.5
    cases_file = sys.argv[4] if len(sys.argv) > 4 else None
    main(sys.argv[1], sys.argv[2], threshold, cases_file)
