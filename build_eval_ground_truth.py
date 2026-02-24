import argparse
import json
from pathlib import Path
from typing import Dict, List


REQUIRED_KEYS = {
    "clip_id",
    "dominant_operation",
    "temporal_segment",
    "anticipated_next_operation",
}


def _load_items(path: Path) -> List[Dict]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        if "items" in payload and isinstance(payload["items"], list):
            payload = payload["items"]
        else:
            raise ValueError("JSON object input is only supported with an 'items' list key")
    if not isinstance(payload, list):
        raise ValueError("Input manifest must be a JSON list")
    return payload


def _valid_item(item: Dict) -> bool:
    if not isinstance(item, dict):
        return False
    return REQUIRED_KEYS.issubset(item.keys())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build eval/ground_truth_u0108_30.json from a normalized clips manifest"
    )
    parser.add_argument("--manifest", type=Path, required=True, help="JSON list with clip metadata")
    parser.add_argument("--subject-prefix", type=str, default="U0108_", help="clip_id prefix filter")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("eval/ground_truth_u0108_30.json"))
    args = parser.parse_args()

    items = _load_items(args.manifest)
    filtered = [x for x in items if _valid_item(x) and str(x.get("clip_id", "")).startswith(args.subject_prefix)]
    filtered.sort(key=lambda x: x["clip_id"])
    selected = filtered[: args.limit]

    if len(selected) < args.limit:
        raise ValueError(
            f"Only found {len(selected)} items with prefix '{args.subject_prefix}'. Need {args.limit}."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2))

    ids_path = args.output.parent / "u0108_selected_clip_ids.txt"
    ids_path.write_text("\n".join(x["clip_id"] for x in selected) + "\n")

    print(f"Wrote {len(selected)} ground-truth items to {args.output}")
    print(f"Wrote selected clip ids to {ids_path}")


if __name__ == "__main__":
    main()
