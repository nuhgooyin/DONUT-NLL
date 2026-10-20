"""Rule-based pair pickers: choose 6 of the 36 combinations for each pair. An initial step
   that should be tested so we can be more informed on which factors (e.g., confidence,
   etc.) are most important regarding pair prediction, before moving onto an ML-based
   approach.

Usage:
    python hand_picker.py <pred.pkl> <pairs.csv> <picker> <out.csv> [--check <gt.pkl>]

Pickers:
    product   standard baseline: top 6 by conf_A x conf_B (this is what MTR does)
    diverse   product score, but each agent's guess may appear at most
              MAX_REPEAT times among the 6 picks, so the picks spread out
    ranksum   rank combinations by rank_A + rank_B (ties broken by product)

Output: one row per pick, 6 per pair:
    scenario_id, agent_a, agent_b, rank, guess_a, guess_b, score

Note that guess_a and guess_b are indices 0-5 into each agent's 6 predictions.

Also note these pickers only use the predictions (no answer key needed). However, the --check flag
can optionally be used for a quick miss-rate check with the answer key. For the more important
metric of mAP, the grader script is needed.

"""

import argparse
import csv
from collections import defaultdict

import numpy as np

from interaction_tests import load_agents, load_pred, hits, CHECKPOINTS
from utils import load_data

MAX_REPEAT = 2


def pick_product(pia, pib):
    score = pia[:, None] * pib[None, :]
    flat = np.argsort(-score.ravel())[:6]
    return [(f // 6, f % 6, score.ravel()[f]) for f in flat]


def pick_diverse(pia, pib, max_repeat=MAX_REPEAT):
    score = pia[:, None] * pib[None, :]
    used_a, used_b, picks = defaultdict(int), defaultdict(int), []
    for f in np.argsort(-score.ravel()):
        i, j = f // 6, f % 6
        if used_a[i] < max_repeat and used_b[j] < max_repeat:
            picks.append((i, j, score[i, j]))
            used_a[i] += 1
            used_b[j] += 1
            if len(picks) == 6:
                break
    return picks


def pick_ranksum(pia, pib):
    ra = np.argsort(np.argsort(-pia))       # 0 = most confident
    rb = np.argsort(np.argsort(-pib))
    product = pia[:, None] * pib[None, :]
    combos = [(ra[i] + rb[j], -product[i, j], i, j)
              for i in range(6) for j in range(6)]
    combos.sort()
    return [(i, j, product[i, j]) for _, _, i, j in combos[:6]]


PICKERS = {'product': pick_product, 'diverse': pick_diverse, 'ranksum': pick_ranksum}


def read_pairs(path):
    pairs = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            pairs[row['scenario_id']].append((row['agent_a'], row['agent_b']))
    return pairs


def run(pred, pairs, picker):
    rows = []
    for sid, scene_pred in pred.items():
        for a_id, b_id in pairs.get(sid, []):
            if a_id not in scene_pred or b_id not in scene_pred:
                continue
            _, pia = load_pred(scene_pred[a_id])
            _, pib = load_pred(scene_pred[b_id])
            for rank, (i, j, s) in enumerate(PICKERS[picker](pia, pib)):
                rows.append({'scenario_id': sid, 'agent_a': a_id, 'agent_b': b_id,
                             'rank': rank, 'guess_a': int(i), 'guess_b': int(j),
                             'score': float(s)})
    return rows


def check(rows, pred, gt):
    """Quick joint miss rate of the picks (same counting as interaction_tests)."""
    picks = defaultdict(list)
    for r in rows:
        picks[(r['scenario_id'], r['agent_a'], r['agent_b'])].append(
            (r['guess_a'], r['guess_b']))
    used, miss = defaultdict(int), defaultdict(int)
    for (sid, a_id, b_id), chosen in picks.items():
        if sid not in gt:
            continue
        agents = load_agents(gt[sid])
        a, b = agents[a_id], agents[b_id]
        pa, _ = load_pred(pred[sid][a_id])
        pb, _ = load_pred(pred[sid][b_id])
        for name, t in CHECKPOINTS.items():
            if not (a['valid'][t] and b['valid'][t]):
                continue
            ha, hb = hits(pa[:, t], a, t), hits(pb[:, t], b, t)
            used[name] += 1
            miss[name] += not any(ha[i] and hb[j] for i, j in chosen)
    print(f'{"checkpoint":<12}{"pairs used":>11}{"joint miss":>12}{"kept":>8}')
    for name in CHECKPOINTS:
        n = max(used[name], 1)
        print(f'{name:<12}{used[name]:>11}{miss[name] / n:>12.3f}{1 - miss[name] / n:>8.1%}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pred')
    ap.add_argument('pairs')
    ap.add_argument('picker', choices=sorted(PICKERS))
    ap.add_argument('out')
    ap.add_argument('--check', metavar='GT', help='answer key, for a quick miss-rate check')
    args = ap.parse_args()

    pred = load_data(args.pred)
    rows = run(pred, read_pairs(args.pairs), args.picker)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['scenario_id', 'agent_a', 'agent_b',
                                          'rank', 'guess_a', 'guess_b', 'score'])
        w.writeheader()
        w.writerows(rows)
    print(f'{args.picker}: wrote {len(rows)} picks for {len(rows) // 6} pairs to {args.out}')
    if args.check:
        check(rows, pred, load_data(args.check))


if __name__ == '__main__':
    main()
