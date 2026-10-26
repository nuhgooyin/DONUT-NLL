"""Score picker output CSVs with joint miss rate and Waymo-style hard mAP.

Usage:
    python joint_grader.py <picks.csv> <mAP_rectangles.pkl> <gt.pkl> \
        --split tune|report [--ranking score|rank]

The CSV must contain six rows per (scenario_id, agent_a, agent_b), with
scenario_id, agent_a, agent_b, rank, guess_a, guess_b, and score columns.
"""

import argparse
import csv
import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

CHECKPOINTS = {"3s": 5, "5s": 9, "8s": 15}
TYPE_NAMES = {1: "vehicle", 2: "pedestrian", 3: "cyclist"}
TYPE_PRIORITY = {"vehicle": 2, "pedestrian": 3, "cyclist": 4}
TRAJECTORY_PRIORITY = {
    "stationary": 0,
    "straight": 1,
    "straight_right": 2,
    "straight_left": 3,
    "right_turn": 4,
    "left_turn": 5,
    "left_u_turn": 6,
    "right_u_turn": 7,
}
TRAJECTORY_BUCKETS = tuple(TRAJECTORY_PRIORITY)


@dataclass(frozen=True)
class Pick:
    scenario_id: str
    agent_a: str
    agent_b: str
    rank: int
    guess_a: int
    guess_b: int
    score: float


def scenario_split(scenario_id: str) -> str:
    """Use the same even-MD5=tune, odd-MD5=report split as EXPERIMENTS.md."""
    digest = hashlib.md5(scenario_id.encode("utf-8")).hexdigest()
    return "tune" if int(digest, 16) % 2 == 0 else "report"


def read_picks(path: Path, split: str, ranking: str) -> dict:
    required = {
        "scenario_id", "agent_a", "agent_b", "rank", "guess_a", "guess_b"
    }
    if ranking == "score":
        required.add("score")

    grouped = defaultdict(list)
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"Picks CSV has no header: {path}")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Picks CSV is missing columns: {', '.join(sorted(missing))}")

        for line_number, row in enumerate(reader, start=2):
            scenario_id = row["scenario_id"].strip()
            if scenario_split(scenario_id) != split:
                continue
            try:
                rank = int(row["rank"])
                guess_a = int(row["guess_a"])
                guess_b = int(row["guess_b"])
                score = float(row["score"]) if ranking == "score" else -float(rank)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid numeric value on CSV line {line_number}") from error
            if not math.isfinite(score):
                raise ValueError(f"Non-finite score on CSV line {line_number}")
            if not (0 <= guess_a < 6 and 0 <= guess_b < 6):
                raise ValueError(
                    f"Guess indices must be 0..5 on CSV line {line_number}"
                )

            pick = Pick(
                scenario_id=scenario_id,
                agent_a=row["agent_a"].strip(),
                agent_b=row["agent_b"].strip(),
                rank=rank,
                guess_a=guess_a,
                guess_b=guess_b,
                score=score,
            )
            grouped[(pick.scenario_id, pick.agent_a, pick.agent_b)].append(pick)

    for pair, picks in grouped.items():
        if len(picks) != 6:
            raise ValueError(f"Expected six picks for {pair}, found {len(picks)}")
        ranks = [pick.rank for pick in picks]
        guesses = [(pick.guess_a, pick.guess_b) for pick in picks]
        if len(set(ranks)) != 6:
            raise ValueError(f"Ranks are not unique for {pair}")
        if len(set(guesses)) != 6:
            raise ValueError(f"Duplicate guess pair for {pair}")
        picks.sort(key=lambda pick: (pick.rank, -pick.score))

    return grouped


def _normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def classify_trajectory(agent: dict) -> str | None:
    """Approximate Waymo's trajectory-shape class from stored 0.5 s GT samples."""
    import numpy as np

    valid_indices = np.flatnonzero(agent["valid"])
    current = agent["cur"]
    if current is None or len(valid_indices) == 0:
        return None

    last_index = int(valid_indices[-1])
    start_heading = float(agent["cur_head"])
    end_heading = float(agent["head"][last_index])
    start_speed = float(agent["speed"])

    end_position = agent["pos"][last_index]
    delta = end_position - current
    cos_heading, sin_heading = math.cos(start_heading), math.sin(start_heading)
    longitudinal = delta[0] * cos_heading + delta[1] * sin_heading
    lateral = -delta[0] * sin_heading + delta[1] * cos_heading
    displacement = float(np.linalg.norm(delta))

    if len(valid_indices) > 1:
        previous_index = int(valid_indices[-2])
        elapsed = (last_index - previous_index) * 0.5
        end_speed = float(
            np.linalg.norm(agent["pos"][last_index] - agent["pos"][previous_index])
            / elapsed
        )
    else:
        end_speed = start_speed

    if max(start_speed, end_speed) < 2.0 and displacement < 3.0:
        return "stationary"

    heading_difference = _normalize_angle(end_heading - start_heading)
    if abs(heading_difference) < math.pi / 6:
        if abs(lateral) < 2.5:
            return "straight"
        return "straight_right" if lateral < 0 else "straight_left"
    if lateral < 0:
        return "right_u_turn" if longitudinal < 0 else "right_turn"
    return "left_u_turn" if longitudinal < 0 else "left_turn"


def pair_bucket(agent_a: dict, agent_b: dict) -> tuple[str, str] | None:
    type_a = TYPE_NAMES.get(agent_a["type"])
    type_b = TYPE_NAMES.get(agent_b["type"])
    trajectory_a = classify_trajectory(agent_a)
    trajectory_b = classify_trajectory(agent_b)
    if None in (type_a, type_b, trajectory_a, trajectory_b):
        return None
    object_type = max((type_a, type_b), key=TYPE_PRIORITY.__getitem__)
    trajectory_type = max(
        (trajectory_a, trajectory_b), key=TRAJECTORY_PRIORITY.__getitem__
    )
    return object_type, trajectory_type


def average_precision(samples: list[tuple[float, bool]], total_positives: int) -> float:
    """VOC-style all-points AP used by Waymo, with false positives first on ties."""
    if total_positives <= 0 or not samples:
        return 0.0
    ordered = sorted(samples, key=lambda item: (-item[0], item[1]))
    true_positives = 0
    precisions = []
    recalls = []
    for index, (_, is_true_positive) in enumerate(ordered, start=1):
        true_positives += bool(is_true_positive)
        precisions.append(true_positives / index)
        recalls.append(true_positives / total_positives)

    recall = [0.0, *recalls, 1.0]
    precision = [0.0, *precisions, 0.0]
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    return sum(
        (recall[index + 1] - recall[index]) * precision[index + 1]
        for index in range(len(recall) - 1)
        if recall[index + 1] != recall[index]
    )


def score_picks(
    picks_by_pair: dict,
    predictions: dict,
    ground_truth: dict,
    split: str,
    load_agents: Callable,
    load_pred: Callable,
    hits: Callable,
) -> tuple[dict, dict, int]:
    misses = {name: [0, 0] for name in CHECKPOINTS}
    map_groups = {
        (object_type, trajectory_type, checkpoint): [[], 0]
        for object_type in TYPE_NAMES.values()
        for trajectory_type in TRAJECTORY_BUCKETS
        for checkpoint in CHECKPOINTS
    }
    scored_pairs = 0

    for (scenario_id, agent_a_id, agent_b_id), picks in picks_by_pair.items():
        if scenario_split(scenario_id) != split:
            continue
        if scenario_id not in predictions or scenario_id not in ground_truth:
            raise ValueError(f"Scenario {scenario_id} is missing from predictions or GT")
        scene_predictions = predictions[scenario_id]
        agents = load_agents(ground_truth[scenario_id])
        if agent_a_id not in scene_predictions or agent_b_id not in scene_predictions:
            raise ValueError(f"Predictions missing an agent for pair {(scenario_id, agent_a_id, agent_b_id)}")
        if agent_a_id not in agents or agent_b_id not in agents:
            raise ValueError(f"Ground truth missing an agent for pair {(scenario_id, agent_a_id, agent_b_id)}")

        agent_a, agent_b = agents[agent_a_id], agents[agent_b_id]
        positions_a, _ = load_pred(scene_predictions[agent_a_id])
        positions_b, _ = load_pred(scene_predictions[agent_b_id])
        bucket = pair_bucket(agent_a, agent_b)
        scored_pairs += 1

        for checkpoint, step in CHECKPOINTS.items():
            if not (agent_a["valid"][step] and agent_b["valid"][step]):
                continue
            hits_a = hits(positions_a[:, step], agent_a, step)
            hits_b = hits(positions_b[:, step], agent_b, step)
            pick_hits = [
                bool(hits_a[pick.guess_a] and hits_b[pick.guess_b]) for pick in picks
            ]

            misses[checkpoint][1] += 1
            misses[checkpoint][0] += not any(pick_hits)
            if bucket is None:
                continue

            object_type, trajectory_type = bucket
            samples, total_positives = map_groups[
                (object_type, trajectory_type, checkpoint)
            ]
            already_found_positive = False
            for pick, is_hit in zip(picks, pick_hits):
                is_true_positive = bool(is_hit and not already_found_positive)
                samples.append((pick.score, is_true_positive))
                already_found_positive |= bool(is_hit)
            map_groups[(object_type, trajectory_type, checkpoint)][1] = (
                total_positives + 1
            )

    return misses, map_groups, scored_pairs


def summarize(misses: dict, map_groups: dict, scored_pairs: int) -> None:
    print(f"Scored pairs: {scored_pairs}")
    print("Miss rate counts any of the six selected pairs as a hit.")
    print("mAP uses global confidence ranking, one TP per pair, and trajectory-shape buckets.")
    print(
        f"{'horizon':<10}{'pairs used':>12}{'mAP pairs':>12}{'joint miss':>13}"
        f"{'vehicle mAP':>14}{'pedestrian mAP':>17}{'cyclist mAP':>14}{'type macro mAP':>16}"
    )
    for checkpoint in CHECKPOINTS:
        miss_count, pair_count = misses[checkpoint]
        map_pair_count = sum(
            map_groups[(object_type, trajectory_type, checkpoint)][1]
            for object_type in TYPE_NAMES.values()
            for trajectory_type in TRAJECTORY_BUCKETS
        )
        per_type = {}
        for object_type in TYPE_NAMES.values():
            bucket_aps = []
            for trajectory_type in TRAJECTORY_BUCKETS:
                samples, positives = map_groups[
                    (object_type, trajectory_type, checkpoint)
                ]
                if positives:
                    bucket_aps.append(average_precision(samples, positives))
            per_type[object_type] = (
                sum(bucket_aps) / len(bucket_aps) if bucket_aps else None
            )
        available = [value for value in per_type.values() if value is not None]
        macro_map = sum(available) / len(available) if available else None

        def format_metric(value):
            return f"{value:.3f}" if value is not None else "n/a"

        print(
            f"{checkpoint:<10}{pair_count:>12}{map_pair_count:>12}"
            f"{miss_count / pair_count if pair_count else float('nan'):>13.3f}"
            f"{format_metric(per_type['vehicle']):>14}"
            f"{format_metric(per_type['pedestrian']):>17}"
            f"{format_metric(per_type['cyclist']):>14}"
            f"{format_metric(macro_map):>16}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("picks", type=Path, help="Picker output CSV")
    parser.add_argument("predictions", type=Path, help="mAP_rectangles prediction pickle")
    parser.add_argument("ground_truth", type=Path, help="Waymo answer-key pickle")
    parser.add_argument("--split", choices=("tune", "report"), required=True)
    parser.add_argument(
        "--ranking",
        choices=("score", "rank"),
        default="score",
        help="Use globally comparable CSV scores for mAP, or rank as a fallback",
    )
    args = parser.parse_args()

    from interaction_tests import hits, load_agents, load_pred
    from utils import load_data

    picks = read_picks(args.picks, args.split, args.ranking)
    if not picks:
        raise SystemExit(f"No picks found for the {args.split} split")

    predictions = load_data(args.predictions)
    ground_truth = load_data(args.ground_truth)
    misses, map_groups, scored_pairs = score_picks(
        picks, predictions, ground_truth, args.split, load_agents, load_pred, hits
    )
    print(f"Split: {args.split} (even MD5=tune, odd MD5=report)")
    print(f"Ranking column: {args.ranking}")
    summarize(misses, map_groups, scored_pairs)


if __name__ == "__main__":
    main()