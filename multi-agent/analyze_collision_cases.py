"""For each predicted vehicle-vehicle collision, check which agent's top guess was wrong.

Usage:
    python analyze_collision_cases.py <predictions.pkl> <gt.pkl> <cases.csv>

cases.csv is the file written by measure_collisions.py. For every case, this compares each
agent's most confident guess with where that agent really was at that checkpoint, and asks:
is the LESS confident agent usually the one whose guess is worse? If so, a simple fix
("when two top guesses collide, demote the less confident one") targets the right guess.
"""
import csv
import pickle
import sys

import numpy as np

from utils import unpack_npy

CHECKPOINTS = {'3s': 5, '5s': 9, '8s': 15}
CLOSE_M = {'3s': 2.0, '5s': 3.6, '8s': 6.0}  # roughly Waymo's longitudinal tolerance per checkpoint


def top_guess_error(pred, gt, scene_id, agent_id, i):
    p = pred[scene_id][agent_id]
    pos = unpack_npy(p['pos'])
    pi = unpack_npy(p['pi']).reshape(-1)
    g = gt[scene_id][agent_id]
    real = unpack_npy(g['eval_pos']).reshape(-1, 2)
    valid = unpack_npy(g['eval_valid']).reshape(-1)
    if not valid[i]:
        return float('nan')
    return float(np.linalg.norm(pos[int(np.argmax(pi))][i] - real[i]))


def main(pred_path, gt_path, cases_path):
    with open(pred_path, 'rb') as f:
        pred = pickle.load(f)
    with open(gt_path, 'rb') as f:
        gt = pickle.load(f)
    with open(cases_path) as f:
        cases = list(csv.DictReader(f))

    n = lower_conf_worse = both_close = neither_close = only_high_close = only_low_close = 0
    for c in cases:
        i = CHECKPOINTS[c['checkpoint']]
        a, b = c['agent_a'], c['agent_b']
        err_a = top_guess_error(pred, gt, c['scenario_id'], a, i)
        err_b = top_guess_error(pred, gt, c['scenario_id'], b, i)
        if np.isnan(err_a) or np.isnan(err_b):
            continue
        n += 1
        # "high" = the more confident agent, "low" = the less confident one
        if float(c['conf_a']) >= float(c['conf_b']):
            err_high, err_low = err_a, err_b
        else:
            err_high, err_low = err_b, err_a
        lower_conf_worse += err_low > err_high
        tol = CLOSE_M[c['checkpoint']]
        high_ok, low_ok = err_high < tol, err_low < tol
        both_close += high_ok and low_ok
        neither_close += (not high_ok) and (not low_ok)
        only_high_close += high_ok and not low_ok
        only_low_close += low_ok and not high_ok

    print(f'Cases analysed: {n}')
    print(f'Less confident agent has the worse guess: {lower_conf_worse} ({100 * lower_conf_worse / max(n, 1):.0f}%)')
    print()
    print('Was each top guess close to the real position? (within ~Waymo tolerance)')
    print(f'  only the MORE confident one was close: {only_high_close}')
    print(f'  only the LESS confident one was close: {only_low_close}')
    print(f'  both close:                            {both_close}')
    print(f'  neither close:                         {neither_close}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
