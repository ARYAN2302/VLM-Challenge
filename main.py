import json
import os
import re
import tempfile
from typing import Any, Dict, Optional

import torch
from decord import VideoReader, cpu
from fastapi import FastAPI, File, HTTPException, UploadFile
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "").strip()
ALLOWED_OPS = {
    "Box Setup",
    "Inner Packing",
    "Tape",
    "Put Items",
    "Pack",
    "Wrap",
    "Label",
    "Final Check",
    "Idle",
    "Unknown",
}


class TemporalSegment(BaseModel):
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)


class PredictionResponse(BaseModel):
    clip_id: str
    dominant_operation: str
    temporal_segment: TemporalSegment
    anticipated_next_operation: str
    confidence: float = Field(ge=0.0, le=1.0)


app = FastAPI(title="Temporal VLM API", version="1.0")
model = None
processor = None


@app.on_event("startup")
async def load_model() -> None:
    global model, processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    model = base_model
    if ADAPTER_PATH:
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, is_trainable=False)
        model = model.merge_and_unload()
    if device == "cpu":
        model.to(device)


def extract_uniform_frames(video_path: str, num_frames: int = 8):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    if total_frames == 0:
        raise ValueError("Uploaded video has zero frames")

    # Avoid index overflow at clip end.
    indices = [min(int(i * total_frames / num_frames), total_frames - 1) for i in range(num_frames)]
    return vr.get_batch(indices).asnumpy(), total_frames


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _normalize_prediction(raw: Dict[str, Any], clip_id: str, total_frames: int) -> PredictionResponse:
    dominant_operation = str(raw.get("dominant_operation", "Unknown"))
    if dominant_operation not in ALLOWED_OPS:
        dominant_operation = "Unknown"

    next_operation = str(raw.get("anticipated_next_operation", "Unknown"))
    if next_operation not in ALLOWED_OPS:
        next_operation = "Unknown"

    segment = raw.get("temporal_segment", {}) if isinstance(raw.get("temporal_segment", {}), dict) else {}
    start = int(segment.get("start_frame", 0))
    end = int(segment.get("end_frame", max(0, total_frames - 1)))

    start = max(0, min(start, total_frames - 1))
    end = max(0, min(end, total_frames - 1))
    if end < start:
        end = start

    confidence = float(raw.get("confidence", 0.5))
    confidence = max(0.0, min(confidence, 1.0))

    return PredictionResponse(
        clip_id=clip_id,
        dominant_operation=dominant_operation,
        temporal_segment=TemporalSegment(start_frame=start, end_frame=end),
        anticipated_next_operation=next_operation,
        confidence=confidence,
    )


def _fallback_prediction(clip_id: str, total_frames: int) -> PredictionResponse:
    end = max(0, total_frames - 1)
    return PredictionResponse(
        clip_id=clip_id,
        dominant_operation="Unknown",
        temporal_segment=TemporalSegment(start_frame=0, end_frame=end),
        anticipated_next_operation="Unknown",
        confidence=0.2,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_video(file: UploadFile = File(...)) -> PredictionResponse:
    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only MP4 uploads are supported")

    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        video_path = tmp.name

    clip_id = os.path.splitext(file.filename)[0]

    try:
        frames, total_frames = extract_uniform_frames(video_path, num_frames=8)

        prompt = (
            "Analyze this 5-second warehouse packaging clip and return only valid JSON with keys: "
            "clip_id, dominant_operation, temporal_segment(start_frame,end_frame), "
            "anticipated_next_operation, confidence. "
            "Use operation labels from: Box Setup, Inner Packing, Tape, Put Items, Pack, Wrap, Label, Final Check, Idle, Unknown."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], videos=[frames], padding=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=196)

        prompt_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, prompt_len:]
        output_text = processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        raw = _extract_json_block(output_text)
        if raw is None:
            return _fallback_prediction(clip_id=clip_id, total_frames=total_frames)

        raw["clip_id"] = clip_id
        return _normalize_prediction(raw=raw, clip_id=clip_id, total_frames=total_frames)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}
