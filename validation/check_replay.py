"""Compare full P1 reruns, excluding wall time and source-report metadata."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads((args.original / "result.json").read_text(encoding="utf-8"))
    replay = json.loads((args.replay / "result.json").read_text(encoding="utf-8"))
    fields = (
        "config_hash",
        "series_sha256",
        "state_digest",
        "rng_state",
        "moves",
        "summary",
        "comparisons",
    )
    checks = {field: original[field] == replay[field] for field in fields}
    checks["completed"] = original["status"] == replay["status"] == "COMPLETED"
    checks["raw_series"] = np.array_equal(
        np.load(args.original / "series.npy", allow_pickle=False),
        np.load(args.replay / "series.npy", allow_pickle=False),
    )
    with np.load(args.original / "final.npz", allow_pickle=False) as left, np.load(
        args.replay / "final.npz", allow_pickle=False
    ) as right:
        checks["checkpoint_members"] = left.files == right.files and all(
            np.array_equal(left[key], right[key]) for key in left.files
        )
    result = {
        "status": "VERIFIED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "series_sha256": original["series_sha256"],
        "state_digest": original["state_digest"],
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
