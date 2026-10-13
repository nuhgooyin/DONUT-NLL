from interaction_tests import load_agents, load_pred, hits, CHECKPOINTS, TYPE_NAMES
from utils import load_data

import csv
import sys
from collections import defaultdict
import numpy as np

FIELDS = ['scenario_id', 'agent_a', 'agent_b', 'pair_type', 'checkpoint',
          'kept_hit', 'any_of_36_hit']

def read_pairs(path):
    """Return {scenario_id: [(agent_a, agent_b), ...]} from the pairs CSV."""
    pairs = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            pairs[row['scenario_id']].append((row['agent_a'], row['agent_b']))
    return pairs


def pair_type(a, b):
    """Order-independent label, e.g. 'pedestrian-vehicle'."""
    names = sorted([TYPE_NAMES.get(a['type'], 'other'),
                    TYPE_NAMES.get(b['type'], 'other')])
    return '-'.join(names)


def collect_rows(pred, gt, pairs):
    rows = []
    for sid, scene_pred in pred.items():
        if sid not in gt or sid not in pairs:
            continue
        agents = load_agents(gt[sid])

        for a_id, b_id in pairs[sid]:
            if a_id not in scene_pred or b_id not in scene_pred:
                continue
            if a_id not in agents or b_id not in agents:
                continue
            a, b = agents[a_id], agents[b_id]
            pa, pia = load_pred(scene_pred[a_id])
            pb, pib = load_pred(scene_pred[b_id])

            # standard picker: rank the 36 combinations by confidence product
            score = pia[:, None] * pib[None, :]          # [6, 6]
            top6 = np.argsort(-score.ravel())[:6]
            label = pair_type(a, b)

            for name, i in CHECKPOINTS.items():
                # skip if either agent has no real position at this checkpoint
                if not (a['valid'][i] and b['valid'][i]):
                    continue
                ha, hb = hits(pa[:, i], a, i), hits(pb[:, i], b, i)
                both = (ha[:, None] & hb[None, :]).ravel()   # [36] pair is right
                rows.append({
                    'scenario_id': sid,
                    'agent_a': a_id,
                    'agent_b': b_id,
                    'pair_type': label,
                    'checkpoint': name,
                    'kept_hit': bool(both[top6].any()),
                    'any_of_36_hit': bool(both.any()),
                })
    return rows


def write_rows(rows, path):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, min_pairs):
    # (pair_type, checkpoint) -> [count, kept hits, any-of-36 hits]
    groups = defaultdict(lambda: [0, 0, 0])
    totals = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        for table, key in ((groups, (r['pair_type'], r['checkpoint'])),
                           (totals, r['checkpoint'])):
            table[key][0] += 1
            table[key][1] += r['kept_hit']
            table[key][2] += r['any_of_36_hit']

    types = sorted({t for t, _ in groups},
                   key=lambda t: -groups[(t, '8s')][0])   # most common first

    header = (f'{"pair type":<24}{"pairs":>7}{"kept":>8}{"among 36":>10}'
              f'{"gap":>8}{"pairs lost":>12}')
    for name in CHECKPOINTS:
        print(f'\nCheckpoint {name}')
        print(header)
        for t in types:
            n, kept, any36 = groups[(t, name)]
            if n == 0:
                continue
            gap = (any36 - kept) / n
            flag = '  (small)' if n < min_pairs else ''
            print(f'{t:<24}{n:>7}{kept / n:>8.1%}{any36 / n:>10.1%}'
                  f'{gap * 100:>7.1f}p{any36 - kept:>12}{flag}')
        n, kept, any36 = totals[name]
        if n:
            print(f'{"all":<24}{n:>7}{kept / n:>8.1%}{any36 / n:>10.1%}'
                  f'{(any36 - kept) / n * 100:>7.1f}p{any36 - kept:>12}')

    print('\nkept       = right pair in the standard picker\'s top 6')
    print('among 36   = right pair anywhere in the 36 combinations')
    print('gap        = among 36 minus kept, in percentage points')
    print('pairs lost = pairs where a right combination existed but was not kept')
    print(f'(small)    = fewer than {min_pairs} pairs; percentages are unreliable')

def main():
    if len(sys.argv) not in (5, 6):
        sys.exit(__doc__)
    pred_path, gt_path, pairs_path, out_path = sys.argv[1:5]
    min_pairs = int(sys.argv[5]) if len(sys.argv) == 6 else 100

    pred, gt = load_data(pred_path), load_data(gt_path)
    rows = collect_rows(pred, gt, read_pairs(pairs_path))
    write_rows(rows, out_path)
    print(f'Wrote {len(rows)} rows to {out_path}')
    summarize(rows, min_pairs)


if __name__ == '__main__':
    main()