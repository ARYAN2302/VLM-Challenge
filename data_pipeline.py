import json
import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import webdataset as wds
try:
    from decord import VideoReader, cpu
    HAS_DECORD = True
except Exception:
    HAS_DECORD = False

# --- CONSTANTS ---
TARGET_RESOLUTION = 336
TARGET_FRAMES = 8
FPS = 25
DATA_DIR = Path("./data")
SAMPLES_DIR = Path("./training_data_samples")
SHARDS_DIR = Path("./shards")

DATA_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
SHARDS_DIR.mkdir(parents=True, exist_ok=True)


# --- 1. OPTICAL FLOW SAMPLING ---
def compute_adaptive_motion_keyframes(
    video_path: str,
    start_frame: int,
    end_frame: int,
    target_keyframes: int = TARGET_FRAMES,
) -> List[int]:
    """Return top-motion frame indices inside [start_frame, end_frame]."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    success, prev_frame = cap.read()
    if not success:
        cap.release()
        return []

    down_dim = (160, 120)
    prev_gray = cv2.cvtColor(cv2.resize(prev_frame, down_dim), cv2.COLOR_BGR2GRAY)

    motion_scores: List[float] = []
    analyzed_indices: List[int] = []
    curr_idx = start_frame + 1

    while curr_idx <= end_frame:
        success, curr_frame = cap.read()
        if not success:
            break

        curr_gray = cv2.cvtColor(cv2.resize(curr_frame, down_dim), cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        motion_scores.append(float(np.mean(mag)))
        analyzed_indices.append(curr_idx)

        prev_gray = curr_gray
        curr_idx += 1

    cap.release()

    if len(analyzed_indices) <= target_keyframes:
        return analyzed_indices

    top_indices = np.argsort(motion_scores)[-target_keyframes:]
    optimal_frames = [analyzed_indices[i] for i in top_indices]
    optimal_frames.sort()
    return optimal_frames


# --- 2. HARDWARE-ACCELERATED EXTRACTION ---
def extract_frames_decord(video_path: str, frame_indices: List[int]) -> np.ndarray:
    """Extract and resize specific frames with decord, then pad/truncate to TARGET_FRAMES."""
    if not HAS_DECORD:
        return extract_frames_opencv(video_path, frame_indices)

    vr = VideoReader(video_path, ctx=cpu(0), width=TARGET_RESOLUTION, height=TARGET_RESOLUTION)
    valid_indices = [idx for idx in frame_indices if 0 <= idx < len(vr)]

    if not valid_indices:
        return np.zeros(
            (TARGET_FRAMES, TARGET_RESOLUTION, TARGET_RESOLUTION, 3), dtype=np.uint8
        )

    frames = vr.get_batch(valid_indices).asnumpy()

    if frames.shape[0] < TARGET_FRAMES:
        last = frames[-1:]
        pad_count = TARGET_FRAMES - frames.shape[0]
        frames = np.concatenate([frames, np.repeat(last, pad_count, axis=0)], axis=0)
    elif frames.shape[0] > TARGET_FRAMES:
        frames = frames[:TARGET_FRAMES]

    return frames


def extract_frames_opencv(video_path: str, frame_indices: List[int]) -> np.ndarray:
    """Fallback extractor when decord is unavailable in local environment."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.resize(frame, (TARGET_RESOLUTION, TARGET_RESOLUTION))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if not frames:
        return np.zeros(
            (TARGET_FRAMES, TARGET_RESOLUTION, TARGET_RESOLUTION, 3), dtype=np.uint8
        )

    arr = np.stack(frames, axis=0)
    if arr.shape[0] < TARGET_FRAMES:
        last = arr[-1:]
        pad_count = TARGET_FRAMES - arr.shape[0]
        arr = np.concatenate([arr, np.repeat(last, pad_count, axis=0)], axis=0)
    elif arr.shape[0] > TARGET_FRAMES:
        arr = arr[:TARGET_FRAMES]
    return arr


# --- 3. WEBDATASET SERIALIZATION ---
def process_and_shard(annotations: List[Dict], video_path: str, shard_prefix: str) -> str:
    """Process clips, extract adaptive frames, and write a tar shard."""
    shard_path = SHARDS_DIR / f"{shard_prefix}.tar"

    print(f"Writing dataset shard to {shard_path}...")
    with wds.TarWriter(str(shard_path), encoder=False) as sink:
        for idx, ann in enumerate(annotations):
            clip_id = ann["clip_id"]
            start_f = int(ann["temporal_segment"]["start_frame"])
            end_f = int(ann["temporal_segment"]["end_frame"])

            optimal_indices = compute_adaptive_motion_keyframes(video_path, start_f, end_f)

            if len(optimal_indices) < TARGET_FRAMES:
                fallback_end = max(start_f, min(end_f, start_f + TARGET_FRAMES - 1))
                optimal_indices = list(range(start_f, fallback_end + 1))

            frames = extract_frames_decord(video_path, optimal_indices)

            if idx < 20:
                sample_path = SAMPLES_DIR / f"{clip_id}_sample.jpg"
                grid = np.concatenate(frames, axis=1)
                cv2.imwrite(str(sample_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

            target_json = {
                "clip_id": clip_id,
                "dominant_operation": ann["dominant_operation"],
                "temporal_segment": ann["temporal_segment"],
                "anticipated_next_operation": ann["anticipated_next_operation"],
            }

            sample = {"__key__": clip_id}
            for i, frame in enumerate(frames):
                ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if not ok:
                    continue
                sample[f"{i}.jpg"] = encoded.tobytes()

            sample["json"] = json.dumps(target_json).encode("utf-8")
            sink.write(sample)

    return str(shard_path)


# --- 4. MOCK EXECUTION FOR ASSIGNMENT DELIVERABLE ---
def create_dummy_video(path: str, duration_sec: int = 5) -> None:
    """Create a synthetic MP4 to validate pipeline before full OpenPack download."""
    if os.path.exists(path):
        return

    print(f"Creating dummy video at {path} for pipeline testing...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, FPS, (640, 480))

    for i in range(FPS * duration_sec):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        x = i % 590
        cv2.rectangle(frame, (x, 200), (x + 50, 250), (255, 255, 255), -1)
        out.write(frame)

    out.release()


if __name__ == "__main__":
    dummy_vid_path = DATA_DIR / "U0101_S0100.mp4"
    create_dummy_video(str(dummy_vid_path), duration_sec=10)

    mock_annotations = []
    for i in range(25):
        mock_annotations.append(
            {
                "clip_id": f"U0101_S0100_t{i:04d}",
                "dominant_operation": "Tape",
                "temporal_segment": {"start_frame": i * 5, "end_frame": (i * 5) + 40},
                "anticipated_next_operation": "Put Items",
            }
        )

    out_shard = process_and_shard(
        annotations=mock_annotations,
        video_path=str(dummy_vid_path),
        shard_prefix="openpack_train_0000",
    )
    print(f"Phase 2 Data Pipeline complete. Samples: {SAMPLES_DIR} | Shard: {out_shard}")
