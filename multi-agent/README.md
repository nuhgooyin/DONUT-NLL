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

`interaction_tests.py`, `measure_collisions.py` and `analyze_collision_cases.py`
import `utils.py` from the TraDiE-policies repo, so copy that file into this
folder before running them. We didn't commit it here to avoid having two
versions.

## Scripts

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
- `measure_collisions.py` checks whether the top predictions of two agents
  overlap more often than the real agents did.
- `analyze_collision_cases.py` looks at those overlap cases and finds which
  agent's prediction was the wrong one.

## Results so far (as of Oct 13, 2026)

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
0.393 at 8 s). Thus, collision avoidance as been dropped as a direction to move forward with.

Reading the sampling code, it can be seen  that the 3,000 samples per agent are drawn separately
for each agent, so sample 17 for one agent has no connection to sample 17 for
another. Building pair predictions directly from samples isn't possible with
the current sampling. The model's 6 modes are different: during prediction,
mode k of each agent only attends to mode k of the other agents, so same-mode
pairs might be consistent with each other. Training scores each agent on its
own, though, so this needs testing.

## Limitations

- These aren't official Waymo joint scores and can't be compared directly with
  MTR or MTR++. That is, we use a subset of the regular validation set instead of the
  official interactive set. So far, we've only measured miss rate (not mAP), and we don't
  average over agent types the way Waymo does. Comparing two pickers on these
  same pairs should still be fair though
- Confidences are from the default window policy (8 s coverage). Based on EXPERIMENTS.md (Run 0004), incorporating 'pi=mean' works better (averaging confidence rather than just taking the 8 s one), so testing can be rerun with that for more current results. 

## Remaining checks

- Rerun the joint test with `pi=mean` predictions for the released model, if we want to be thorough.
- Measure the gap with joint mAP, not just miss rate (to do this we need to write a grader script, more info below).
- Break the gap down by pair type (vehicle-vehicle, vehicle-pedestrian, etc.).
- Test whether same-mode pairs are good pair predictions.

## Future work

1. Grader: one script that scores any set of 6 pair predictions with joint miss rate and joint mAP, averaged the way Waymo does. Scenes are split into a tune half and a report half using the group's existing split from EXPERIMENTS.md (even MD5 hash of the scenario ID = tune, odd = report). We develop on the tune half and only report numbers from the report half.
2. Hand-made picker: score the 36 combinations with rules we write ourselves, for example penalising pairs where both agents go first through the same spot.
3. Learned picker: a small model that predicts which combinations are correct from features such as both confidences, the distance between the two paths, whether they cross, and whether they come from the same mode. Trained on the tune half and reported on the report half.
4. Improvements: remove features one at a time to see which ones matter.
5. Final write-up: Report findings

If possible, we should also run on Waymo's official
interactive validation set for a direct comparison with MTR/MTR++.

## Reproducing Main Tests

```bash
python count_interesting.py /scratch/nuhgoyin/DONUT-NLL/data/waymo/processed/val waymo_pairs.csv

python interaction_tests.py joint /scratch/nuhgoyin/DONUT-NLL/samples/tune/full_default/merged/mAP_rectangles.pkl /scratch/nuhgoyin/DONUT-NLL/samples/gt_val.pkl waymo_pairs.csv
```
