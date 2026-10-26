import csv
import tempfile
import unittest
from pathlib import Path

from joint_grader import average_precision, read_picks, scenario_split


class JointGraderTests(unittest.TestCase):
    def test_average_precision_rewards_correct_prediction_rank(self):
        self.assertAlmostEqual(average_precision([(0.9, True), (0.1, False)], 1), 1.0)
        self.assertAlmostEqual(average_precision([(0.9, False), (0.1, True)], 1), 0.5)

    def test_average_precision_counts_one_true_positive_per_query(self):
        self.assertAlmostEqual(average_precision([(0.9, True), (0.8, False)], 1), 1.0)

    def test_split_is_deterministic_and_valid(self):
        first = scenario_split("scenario-123")
        self.assertIn(first, {"tune", "report"})
        self.assertEqual(first, scenario_split("scenario-123"))

    def test_read_picks_requires_six_unique_combinations(self):
        split_id = next(
            f"scene-{index}"
            for index in range(100)
            if scenario_split(f"scene-{index}") == "report"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "picks.csv"
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "scenario_id", "agent_a", "agent_b", "rank",
                        "guess_a", "guess_b", "score",
                    ],
                )
                writer.writeheader()
                for rank in range(6):
                    writer.writerow({
                        "scenario_id": split_id,
                        "agent_a": "a",
                        "agent_b": "b",
                        "rank": rank,
                        "guess_a": rank,
                        "guess_b": 0,
                        "score": 1.0 / (rank + 1),
                    })
            picks = read_picks(path, "report", "score")
        self.assertEqual(len(picks[(split_id, "a", "b")]), 6)

    def test_read_picks_rejects_fewer_than_six_rows(self):
        split_id = next(
            f"scene-{index}"
            for index in range(100)
            if scenario_split(f"scene-{index}") == "report"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "picks.csv"
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "scenario_id", "agent_a", "agent_b", "rank",
                        "guess_a", "guess_b", "score",
                    ],
                )
                writer.writeheader()
                for rank in range(5):
                    writer.writerow({
                        "scenario_id": split_id,
                        "agent_a": "a",
                        "agent_b": "b",
                        "rank": rank,
                        "guess_a": rank,
                        "guess_b": 0,
                        "score": 1.0 / (rank + 1),
                    })
            with self.assertRaisesRegex(ValueError, "Expected six picks"):
                read_picks(path, "report", "score")


if __name__ == "__main__":
    unittest.main()