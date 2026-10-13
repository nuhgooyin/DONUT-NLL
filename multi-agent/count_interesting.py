"""Count how many scenes carry Waymo's own 'objects of interest' pair, and save the pairs.

Usage (run with Python that has torch installed, e.g. the project's environment):
    python count_interesting.py <processed validation folder> [pairs_out.csv]

The processed folder holds one scenario_<id>.pkl per scene (stage 1 output), e.g.
/scratch/nuhgoyin/DONUT-NLL/data/waymo/processed/val

For every scene it reads `interesting_mask` (copied from Waymo's objects_of_interest)
and `category` (1 = an agent Waymo asks us to predict). It prints how many scenes have
0, 1, 2, ... marked agents, and saves every scene with exactly two marked agents to the
CSV, which `interaction_tests.py joint ... pairs.csv` can read.
"""
import csv
import os
import pickle
import sys
from collections import Counter


def main(folder, out_csv=None):
    files = sorted(f for f in os.listdir(folder) if f.endswith('.pkl'))
    counts = Counter()
    pairs, both_predicted = [], 0
    for k, name in enumerate(files):
        with open(os.path.join(folder, name), 'rb') as f:
            scene = pickle.load(f)
        agent = scene['agent']
        marked = [i for i, v in enumerate(agent['interesting_mask'].tolist()) if v]
        counts[len(marked)] += 1
        if len(marked) == 2:
            a, b = marked
            category = agent['category'].tolist()
            predicted = category[a] > 0 and category[b] > 0
            both_predicted += predicted
            pairs.append([scene['scenario_id'], agent['id'][a], agent['id'][b], int(predicted)])
        if (k + 1) % 5000 == 0:
            print(f'  ...{k + 1} of {len(files)} scenes read', flush=True)

    total = len(files)
    print(f'Scenes read: {total}')
    for n in sorted(counts):
        print(f'  scenes with {n} marked agent(s): {counts[n]} ({100 * counts[n] / max(total, 1):.1f}%)')
    print(f'Scenes with exactly 2 marked agents: {len(pairs)}')
    print(f'  ...of which both agents are also ones we predict: {both_predicted}')

    if out_csv:
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['scenario_id', 'agent_a', 'agent_b', 'both_predicted'])
            w.writerows(pairs)
        print(f'Saved the pairs to {out_csv}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
