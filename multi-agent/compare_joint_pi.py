"""Run the joint interaction test on default and pi=mean predictions.

Run this beside the existing interaction_tests.py on the cluster. The two
prediction pickles should come from the same sampled futures, with only the
window-policy confidence setting changed.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def existing_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"{label} does not exist or is not a file: {resolved}")
    return resolved


def run_joint_test(
    label: str,
    prediction_file: Path,
    answer_key: Path,
    pairs_csv: Path,
    joint_script: Path,
    working_directory: Path,
) -> tuple[int, str]:
    command = [
        sys.executable,
        str(joint_script),
        "joint",
        str(prediction_file),
        str(answer_key),
        str(pairs_csv),
    ]
    print(f"\n===== {label} =====", flush=True)
    print(f"Command: {' '.join(command)}", flush=True)

    process = subprocess.Popen(
        command,
        cwd=working_directory,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    output_lines = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        output_lines.append(line)
    return_code = process.wait()
    print(f"Exit code: {return_code}")
    return return_code, "".join(output_lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare joint top-six results for default and pi=mean predictions "
            "using the existing interaction_tests.py scorer."
        )
    )
    parser.add_argument("--default", required=True, type=Path, help="Default mAP_rectangles prediction pickle")
    parser.add_argument("--pimean", required=True, type=Path, help="pi=mean mAP_rectangles prediction pickle")
    parser.add_argument("--gt", required=True, type=Path, help="Waymo answer-key pickle")
    parser.add_argument("--pairs", required=True, type=Path, help="Waymo-marked pairs CSV")
    parser.add_argument(
        "--joint-script",
        type=Path,
        default=Path("interaction_tests.py"),
        help="Path to the existing interaction_tests.py (default: ./interaction_tests.py)",
    )
    parser.add_argument(
        "--working-directory",
        type=Path,
        help="Directory to run the scorer from (default: the scorer's directory)",
    )
    parser.add_argument("--report", type=Path, help="Optional path for a combined text report")
    args = parser.parse_args()

    default_predictions = existing_file(args.default, "Default predictions")
    pimean_predictions = existing_file(args.pimean, "pi=mean predictions")
    answer_key = existing_file(args.gt, "Answer key")
    pairs_csv = existing_file(args.pairs, "Pairs CSV")
    joint_script = existing_file(args.joint_script, "Joint test script")

    if default_predictions == pimean_predictions:
        raise SystemExit("--default and --pimean must refer to different prediction files")

    working_directory = (
        args.working_directory.expanduser().resolve()
        if args.working_directory
        else joint_script.parent
    )
    if not working_directory.is_dir():
        raise SystemExit(f"Working directory does not exist: {working_directory}")

    runs = [
        run_joint_test(
            "DEFAULT WINDOW POLICY",
            default_predictions,
            answer_key,
            pairs_csv,
            joint_script,
            working_directory,
        ),
        run_joint_test(
            "WINDOW POLICY, pi=mean",
            pimean_predictions,
            answer_key,
            pairs_csv,
            joint_script,
            working_directory,
        ),
    ]

    report_text = "\n".join(
        [
            "Matched joint interaction test",
            f"Default predictions: {default_predictions}",
            f"pi=mean predictions: {pimean_predictions}",
            f"Answer key: {answer_key}",
            f"Pairs CSV: {pairs_csv}",
            f"Joint test script: {joint_script}",
            f"Working directory: {working_directory}",
            "",
            "===== DEFAULT WINDOW POLICY =====",
            runs[0][1],
            f"Exit code: {runs[0][0]}",
            "",
            "===== WINDOW POLICY, pi=mean =====",
            runs[1][1],
            f"Exit code: {runs[1][0]}",
        ]
    )
    if args.report:
        report_path = args.report.expanduser().resolve()
        if not report_path.parent.is_dir():
            raise SystemExit(f"Report directory does not exist: {report_path.parent}")
        report_path.write_text(report_text, encoding="utf-8")
        print(f"Combined report saved to {report_path}")

    return int(any(return_code != 0 for return_code, _ in runs))


if __name__ == "__main__":
    raise SystemExit(main())