"""Stage A measure-first tests for the multi-agent work.

Usage (all read existing files only; no GPU needed):
    python interaction_tests.py crowding <predictions.pkl> <gt.pkl>   # A1
    python interaction_tests.py radius   <gt.pkl>                     # A2
    python interaction_tests.py joint    <predictions.pkl> <gt.pkl> [pairs.csv]   # A3
        (without pairs.csv: homemade pairs; with it: e.g. Waymo's own pairs from count_interesting.py)

<predictions.pkl> is the window-policy file (…mAP_rectangles.pkl).

Notes:
- gt.pkl has no "current" position, so it is estimated as the 0.5 s position minus
  0.5 s x current velocity.
- "Hit" uses Waymo's rule: the guess must be inside an oriented box around the true
  position (lateral/longitudinal tolerances per checkpoint, shrunk for slow agents).
"""
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np

from utils import load_data, unpack_npy

CHECKPOINTS = {'3s': 5, '5s': 9, '8s': 15}
TOL = {5: (1.0, 2.0), 9: (1.8, 3.6), 15: (3.0, 6.0)}  # (lateral, longitudinal) metres
TYPE_NAMES = {1: 'vehicle', 2: 'pedestrian', 3: 'cyclist'}


def load_agents(scene_gt):
    """Return dict agent_id -> info for every agent in the scene."""
    agents = {}
    for aid, g in scene_gt.items():
        pos = unpack_npy(g['eval_pos']).reshape(-1, 2).astype(np.float64)
        valid = unpack_npy(g['eval_valid']).reshape(-1).astype(bool)
        vel = unpack_npy(g['cur_vel']).reshape(2).astype(np.float64)
        cur = pos[0] - 0.5 * vel if valid[0] else None
        agents[aid] = {
            'type': int(unpack_npy(g['agent_type'])),
            'predicted': int(unpack_npy(g['category'])) == 1,
            'pos': pos, 'valid': valid,
            'head': unpack_npy(g['eval_head']).reshape(-1).astype(np.float64),
            'speed': float(np.hypot(*vel)), 'cur': cur,
        }
    return agents


def load_pred(p):
    return (unpack_npy(p['pos']).astype(np.float64),          # [6, 16, 2]
            unpack_npy(p['pi']).reshape(-1).astype(np.float64))  # [6]


def speed_scale(v):
    if v < 1.4:
        return 0.5
    if v > 11.0:
        return 1.0
    return 0.5 + 0.5 * (v - 1.4) / (11.0 - 1.4)


def hits(guesses, agent, i):
    """Boolean array: which guesses ([K, 2] positions at step i) are inside Waymo's box."""
    lat, lon = TOL[i]
    s = speed_scale(agent['speed'])
    d = guesses - agent['pos'][i]
    c, sn = np.cos(agent['head'][i]), np.sin(agent['head'][i])
    d_lon = d[:, 0] * c + d[:, 1] * sn
    d_lat = -d[:, 0] * sn + d[:, 1] * c
    return (np.abs(d_lon) < lon * s) & (np.abs(d_lat) < lat * s)


def scene_arrays(agents):
    ids = list(agents)
    cur = np.array([agents[k]['cur'] if agents[k]['cur'] is not None else [np.nan, np.nan]
                    for k in ids])
    pos = np.stack([agents[k]['pos'] for k in ids])      # [M, 16, 2]
    valid = np.stack([agents[k]['valid'] for k in ids])  # [M, 16]
    return ids, cur, pos, valid


TIME_OK = np.abs(np.arange(16)[:, None] - np.arange(16)[None, :]) <= 6  # within 3 s


def interacting_mask(i_idx, pos, valid, dist=5.0):
    """For agent index i_idx, boolean [M]: whose real path comes within `dist` m of
    agent i's path at times no more than 3 s apart."""
    d = np.linalg.norm(pos[i_idx][None, :, None, :] - pos[:, None, :, :], axis=-1)  # [M,16,16]
    ok = valid[i_idx][None, :, None] & valid[:, None, :] & TIME_OK[None]
    out = ((d < dist) & ok).any(axis=(1, 2))
    out[i_idx] = False
    return out


def crowd_bucket(n):
    if n == 0:
        return '0'
    if n <= 2:
        return '1-2'
    if n <= 5:
        return '3-5'
    return '6+'


def crowding(pred_path, gt_path):
    pred, gt = load_data(pred_path), load_data(gt_path)
    stats = defaultdict(lambda: [0, 0.0, 0, 0.0, 0.0])  # n, speed sum, misses, top err sum, minFDE sum
    vstats = defaultdict(lambda: [0, 0, 0.0])            # vehicles by (speed band, crowd): n, misses, top err
    i = CHECKPOINTS['8s']
    for sid, scene_pred in pred.items():
        agents = load_agents(gt[sid])
        ids, cur, _, _ = scene_arrays(agents)
        for aid, p in scene_pred.items():
            a = agents[aid]
            if a['cur'] is None or not a['valid'][i]:
                continue
            dists = np.linalg.norm(cur - a['cur'], axis=1)
            n20 = int(np.sum(dists < 20.0)) - 1  # minus itself (distance 0)
            pos, pi = load_pred(p)
            g8 = pos[:, i]
            err = np.linalg.norm(g8 - a['pos'][i], axis=1)
            key = (TYPE_NAMES.get(a['type'], 'other'), crowd_bucket(n20))
            s = stats[key]
            s[0] += 1
            s[1] += a['speed']
            s[2] += not hits(g8, a, i).any()
            s[3] += err[np.argmax(pi)]
            s[4] += err.min()
            if a['type'] == 1:
                band = 'slow <2' if a['speed'] < 2 else ('mid 2-8' if a['speed'] < 8 else 'fast 8+')
                v = vstats[(band, crowd_bucket(n20))]
                v[0] += 1
                v[1] += not hits(g8, a, i).any()
                v[2] += err[np.argmax(pi)]
    print('A1: performance at 8 s by number of other agents within 20 m (window policy)')
    print(f'{"type":<12}{"nearby":>8}{"agents":>9}{"avg speed":>11}{"miss rate":>11}'
          f'{"top err m":>11}{"minFDE m":>10}')
    order = ['0', '1-2', '3-5', '6+']
    for t in ['vehicle', 'pedestrian', 'cyclist']:
        for b in order:
            n, sp, miss, te, mf = stats.get((t, b), [0, 0, 0, 0, 0])
            if n:
                print(f'{t:<12}{b:>8}{n:>9}{sp / n:>11.1f}{miss / n:>11.3f}{te / n:>11.2f}{mf / n:>10.2f}')
    print('\nNote: compare rows within a type; check whether avg speed differs a lot between rows.')
    print('\nVehicles only, split by speed (m/s), so crowding is compared at similar speeds:')
    print(f'{"speed":<10}{"nearby":>8}{"agents":>9}{"miss rate":>11}{"top err m":>11}')
    for band in ['slow <2', 'mid 2-8', 'fast 8+']:
        for b in order:
            n, miss, te = vstats.get((band, b), [0, 0, 0])
            if n:
                print(f'{band:<10}{b:>8}{n:>9}{miss / n:>11.3f}{te / n:>11.2f}')


def interacting(a, b, dist=5.0, max_dt_steps=6):
    """True if a's and b's real paths come within `dist` m of each other at times
    no more than max_dt_steps (0.5 s each) apart."""
    pa, pb = a['pos'][a['valid']], b['pos'][b['valid']]
    ta, tb = np.flatnonzero(a['valid']), np.flatnonzero(b['valid'])
    if len(pa) == 0 or len(pb) == 0:
        return False
    d = np.linalg.norm(pa[:, None] - pb[None], axis=-1)
    close_t = np.abs(ta[:, None] - tb[None]) <= max_dt_steps
    return bool((d[close_t] < dist).any())


def radius(gt_path):
    gt = load_data(gt_path)
    start_d = []
    min_same_time = []
    for sid, scene_gt in gt.items():
        agents = load_agents(scene_gt)
        ids, cur, pos, valid = scene_arrays(agents)
        for k, aid in enumerate(ids):
            a = agents[aid]
            if not a['predicted'] or a['cur'] is None:
                continue
            partners = interacting_mask(k, pos, valid) & ~np.isnan(cur[:, 0])
            start_d.extend(np.linalg.norm(cur[partners] - a['cur'], axis=1).tolist())
            for j in np.flatnonzero(partners):
                both = valid[k] & valid[j]
                d_t = np.linalg.norm(pos[k][both] - pos[j][both], axis=1)
                d0 = np.linalg.norm(cur[j] - a['cur'])
                min_same_time.append(min(d0, d_t.min()) if len(d_t) else d0)
    d = np.asarray(start_d)
    print('A2: starting distance between predicted agents and the agents they really interact with')
    print(f'Interacting pairs found: {len(d)}')
    if len(d):
        for r in (20, 30, 50, 75, 100):
            print(f'  starting more than {r:>3} m apart: {(d > r).sum():>7} ({100 * (d > r).mean():.2f}%)')
        print(f'  median starting distance: {np.median(d):.1f} m, 99th percentile: {np.percentile(d, 99):.1f} m')
    m = np.asarray(min_same_time)
    if len(m):
        print('\nClosest the two get AT THE SAME MOMENT (the model checks the 50 m radius at each')
        print('predicted time step, not only at the start):')
        for r in (30, 50, 75):
            print(f'  never within {r:>3} m of each other: {(m > r).sum():>7} ({100 * (m > r).mean():.2f}%)')
    print('\nThe model only "sees" agents within 50 m (--dec_radius_s 50).')


def joint(pred_path, gt_path, pairs_csv=None, collide_m=1.5, collide_weight=0.1):
    pred, gt = load_data(pred_path), load_data(gt_path)
    given = None
    if pairs_csv:  # use a fixed list of pairs (e.g. Waymo's own objects_of_interest)
        import csv
        given = defaultdict(list)
        with open(pairs_csv) as f:
            for row in csv.DictReader(f):
                given[row['scenario_id']].append((row['agent_a'], row['agent_b']))
    n_pairs = 0
    n_valid = defaultdict(int)  # pairs where both agents' true positions exist at that checkpoint
    miss = {k: defaultdict(int) for k in ('top6', 'top6_collision_aware', 'best_of_36')}
    marg_miss = defaultdict(int)
    for sid, scene_pred in pred.items():
        if sid not in gt:
            continue
        agents = load_agents(gt[sid])
        ids = [a for a in scene_pred if a in agents]
        if given is not None:
            candidates = [(x, y) for x, y in given.get(sid, []) if x in ids and y in ids]
        else:
            candidates = [(x, y) for x, y in combinations(ids, 2)
                          if interacting(agents[x], agents[y])]
        for a_id, b_id in candidates:
            a, b = agents[a_id], agents[b_id]
            pa, pia = load_pred(scene_pred[a_id])
            pb, pib = load_pred(scene_pred[b_id])
            score = pia[:, None] * pib[None, :]              # [6, 6] pair scores
            # collision-aware: down-weight pairs whose guesses overlap at any checkpoint
            close = np.zeros((6, 6), dtype=bool)
            for i in CHECKPOINTS.values():
                close |= np.linalg.norm(pa[:, None, i] - pb[None, :, i], axis=-1) < collide_m
            score_ca = np.where(close, score * collide_weight, score)
            top6 = np.argsort(-score.ravel())[:6]
            top6_ca = np.argsort(-score_ca.ravel())[:6]
            n_pairs += 1
            for name, i in CHECKPOINTS.items():
                if not (a['valid'][i] and b['valid'][i]):
                    continue
                n_valid[name] += 1
                ha, hb = hits(pa[:, i], a, i), hits(pb[:, i], b, i)
                both = (ha[:, None] & hb[None, :]).ravel()   # [36] pair is right
                miss['top6'][name] += not both[top6].any()
                miss['top6_collision_aware'][name] += not both[top6_ca].any()
                miss['best_of_36'][name] += not both.any()
                marg_miss[name] += (not ha.any()) + (not hb.any())
    source = f'pairs from {pairs_csv}' if pairs_csv else 'homemade pairs (paths within 5 m, within 3 s)'
    print(f'A3: joint miss rate on interacting pairs ({source})')
    print(f'Pairs: {n_pairs}')
    print(f'{"checkpoint":<12}{"pairs used":>11}{"top-6 pairs":>13}{"top-6 coll.-aware":>19}{"best of 36":>12}{"single-agent miss":>19}')
    for name in CHECKPOINTS:
        n = max(n_valid[name], 1)
        print(f'{name:<12}{n_valid[name]:>11}', end='')
        print(f'{miss["top6"][name] / n:>13.3f}{miss["top6_collision_aware"][name] / n:>19.3f}'
              f'{miss["best_of_36"][name] / n:>12.3f}{marg_miss[name] / (2 * n):>19.3f}')
    print('\nGap between "top-6" and "best of 36" = room for a better pair-picking policy.')
    print('"best of 36" itself = how often no combination is right (only model changes can fix).')


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'crowding':
        crowding(sys.argv[2], sys.argv[3])
    elif mode == 'radius':
        radius(sys.argv[2])
    elif mode == 'joint':
        joint(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    else:
        raise SystemExit('mode must be crowding, radius or joint')
