# Multi-agent Interaction

This folder holds our current work on Waymo's joint prediction task. In the joint task,
two interacting agents are predicted together: you submit 6 pair predictions,
and a pair only counts as correct if both agents are inside Waymo's miss box.

DONUT-NLL is a single-agent model, so the question we started with was how well
its predictions work when combined into pairs. The standard way to do this (used
by MTR) is to take each agent's 6 predictions, form all 36 combinations, and
keep the 6 with the highest product of confidences.

## Setup

Run on Vulcan inside an `salloc` job:

```bash
module load StdEnv/2023 python/3.11 scipy-stack
source ~/envs/donut/bin/activate
```

All scripts except `count_interesting.py` import `utils.py` from the TraDiE-policies repo, so copy that file into this folder before running them. We didn't commit it here to avoid having two versions.

## Scripts

- `measure_collisions.py` (earlier check, not used going forward) checks
  whether the top predictions of two agents overlap more often than the real
  agents did.
- `analyze_collision_cases.py` (earlier check, not used going forward) looks
  at those overlap cases and finds which agent's prediction was the wrong one.
- `count_interesting.py` reads the processed validation scenes and counts how
  many have agents marked by Waymo as interacting (`interesting_mask`, which
  comes from Waymo's `objects_of_interest` field). Scenes with exactly two
  marked agents are saved to a CSV, which we use as our pairs.
- `interaction_tests.py` runs one of three tests, picked by the word you type
  after the file name (`joint`, `crowding` or `radius`). `joint` is the main
  one: given a prediction file, the answer key and optionally a pairs CSV, it
  checks for each pair whether the top 6 combinations contain a correct one,
  and whether any of the 36 do. Without a CSV it falls back to our own pairs
  (agents whose real paths came within 5 m of each other). `crowding` and
  `radius` are earlier single-agent checks.
- `compare_joint_pi.py` runs the joint test on the default and `pi=mean`
  prediction files and shows both results.
- `pair_breakdown.py` repeats the joint test but saves one row per pair and
  checkpoint, then prints the results grouped by pair type
  (vehicle-vehicle, pedestrian-vehicle, etc.).
- `initial_pickers.py` chooses 6 of the 36 combinations for each pair using a
  simple rule and saves the picks to a CSV for the grader. `--check` gives a quick 
  joint miss rate. This allows us to be more informed before attempting to
  enact a machine learning approach for pair prediction.

## Results so far (as of Oct 20, 2026)

All results use the released DONUT-NLL checkpoint with the default window
policy (`samples/tune/full_default/merged/mAP_rectangles.pkl`) and the answer
key `samples/gt_val.pkl`.

14,573 of the 44,097 validation scenes (33%) have exactly two
Waymo-marked agents, and in all of them both agents are ones we predict. 29,497
scenes have none and 27 have one.

Results of the joint test on the Waymo-marked pairs:

| Checkpoint | Pairs used | Correct pair in top 6 | Correct pair among all 36 | Gap |
|---|---|---|---|---|
| 3 s | 14,543 | 80.4% | 91.9% | 11.5 pts |
| 5 s | 14,497 | 70.5% | 86.4% | 15.9 pts |
| 8 s | 13,903 | 60.5% | 81.0% | 20.5 pts |

"Pairs used" leaves out pairs where an agent has no real position at that
checkpoint. On our own pairs (88,501 pairs, 64,324 used at 8 s) the numbers are
close: 61.8% vs 83.6% at 8 s.

So at 8 s the standard method throws away a correct pair prediction for about 1
in 5 pairs. That gap is what we're trying to recover with a better way of
choosing the 6 pairs. It shows how much room there is, not how much a picker can
actually get back, since the correct combination is only obvious once you know
the real future.

We also checked whether predictions collide. Predicted agents overlap about as
often as real ones do, and re-ranking pairs to avoid overlaps barely changes the joint miss rate (0.395 to
0.393 at 8 s). Thus, collision avoidance has been dropped as a direction to move forward with.

Reading the sampling code, it can be seen that the 3,000 samples per agent are drawn separately
for each agent, so sample 17 for one agent has no connection to sample 17 for
another. Building pair predictions directly from samples isn't possible with
the current sampling. The model's 6 modes are different: during prediction,
mode k of each agent only attends to mode k of the other agents, so same-mode
pairs might be consistent with each other. Training scores each agent on its
own, though, so this needs testing.

We also reran the joint test with `pi=mean` confidences (Experiment 0004's predictions). The results were very close to the default confidences' 60.5% kept vs. 81.0% among all 36 at 8 s, so this confidence setting does not affect the gap very much.

Breaking the joint test down by pair type (`pair_breakdown.py`), at 8 s:

| Pair type | Pairs | Kept | Among all 36 | Gap | Pairs lost |
|---|---|---|---|---|---|
| vehicle-vehicle | 10,913 | 62.2% | 82.9% | 20.8 pts | 2,269 |
| pedestrian-vehicle | 2,001 | 59.6% | 76.8% | 17.2 pts | 344 |
| cyclist-vehicle | 979 | 44.8% | 68.2% | 23.4 pts | 229 |

It can be seen that vehicle-vehicle pairs make up about 80% of the pairs
where a correct combination existed but wasn't kept, so that's
where a better picker has the most to gain. Cyclist-vehicle pairs have the
largest gap (23.4 points), but there are only 979 of them, so they account for
few of the lost pairs. We left out cyclist-cyclist (10 pairs) and
pedestrian-pedestrian (3 pairs), since with so few pairs one result changes
the percentage a lot. The totals across all types match the main joint test
above.

## Limitations

- These aren't official Waymo joint scores and can't be compared directly with
  MTR or MTR++. That is, we use a subset of the regular validation set instead of the
  official interactive set. So far, we've only measured miss rate (not mAP), and we don't
  average over agent types the way Waymo does. Comparing two pickers on these
  same pairs should still be fair though.

## Task List

Note: Tasks 1 and 2 are being worked on concurrently.

1. Grader **(in progress)**: one script that scores any set of 6 pair predictions with joint miss rate and joint mAP, averaged the way Waymo does. Scenes are split into a tune half and a report half using the group's existing split from EXPERIMENTS.md (even MD5 hash of the scenario ID = tune, odd = report). We develop on the tune half and only report numbers from the report half. Once it works, the first step is to remeasure the gap with joint mAP: score the `product` baseline picker from `initial_pickers.py`, and compare it with a picker that cheats by using the answer key to always keep a correct pair when one exists (the best possible score).
2. Hand-made picker **(in progress)**: score the 36 combinations with rules we write ourselves. A first version is in `initial_pickers.py`, with three rules that only use confidences (product baseline, diverse, ranksum). Planned next are rules that use the predicted paths, for example favouring pairs where one agent clearly reaches a crossing point before the other. Another idea to test could be to pair the same mode for both agents (mode 1 with mode 1, and so on), since DONUT-NLL predicts each mode for all agents together.
3. Learned picker: a small model that predicts which combinations are correct from features such as both confidences, the distance between the two paths, whether they cross, and whether they come from the same mode. Trained on the tune half and reported on the report half.
4. Improvements: remove features one at a time to see which ones matter.
5. Final write-up: Report findings.

If possible, we should also run on Waymo's official
interactive validation set for a direct comparison with MTR/MTR++.

## Reproducing Main Tests

```bash
python count_interesting.py /scratch/nuhgoyin/DONUT-NLL/data/waymo/processed/val waymo_pairs.csv

python interaction_tests.py joint /scratch/nuhgoyin/DONUT-NLL/samples/tune/full_default/merged/mAP_rectangles.pkl /scratch/nuhgoyin/DONUT-NLL/samples/gt_val.pkl waymo_pairs.csv

python pair_breakdown.py /scratch/nuhgoyin/DONUT-NLL/samples/tune/full_default/merged/mAP_rectangles.pkl /scratch/nuhgoyin/DONUT-NLL/samples/gt_val.pkl waymo_pairs.csv pair_rows.csv
```
