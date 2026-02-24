import argparse
import json
from pathlib import Path
from typing import Dict, List

import requests


def _predict_one(endpoint: str, clip_path: Path, timeout: int) -> Dict:
    with clip_path.open("rb") as f:
        files = {"file": (clip_path.name, f, "video/mp4")}
        resp = requests.post(endpoint, files=files, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _fallback_prediction(clip_id: str, error: str) -> Dict:
    return {
        "clip_id": clip_id,
        "dominant_operation": "Unknown",
        "temporal_segment": {"start_frame": 0, "end_frame": 0},
        "anticipated_next_operation": "Unknown",
        "confidence": 0.0,
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch call FastAPI /predict on a directory of clips")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--subject-prefix", default="U0108_")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    if not args.clips_dir.exists():
        raise FileNotFoundError(f"Clips dir does not exist: {args.clips_dir}")

    all_mp4 = sorted(
        [p for p in args.clips_dir.glob("*.mp4") if p.stem.startswith(args.subject_prefix)],
        key=lambda p: p.stem,
    )
    selected = all_mp4[: args.limit]

    if len(selected) < args.limit:
        raise ValueError(
            f"Only found {len(selected)} clips with prefix '{args.subject_prefix}'. Need {args.limit}."
        )

    predictions: List[Dict] = []
    for idx, clip_path in enumerate(selected, start=1):
        clip_id = clip_path.stem
        try:
            pred = _predict_one(args.endpoint, clip_path, timeout=args.timeout)
            if "clip_id" not in pred:
                pred["clip_id"] = clip_id
            predictions.append(pred)
            print(f"[{idx}/{len(selected)}] OK {clip_path.name}")
        except Exception as exc:
            predictions.append(_fallback_prediction(clip_id, str(exc)))
            print(f"[{idx}/{len(selected)}] FAIL {clip_path.name}: {exc}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions, indent=2))
    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
