# AI Development Log

## Phase 1: Base API Deployment
- Tool Used: Codex (GPT-5 coding agent)
- Prompt/Request: Build Phase 1 FastAPI service for Qwen2.5-VL-2B with `/predict` video upload endpoint and Docker deployment files.
- Accepted Output:
  - `main.py` FastAPI app with Qwen2.5-VL model loading, MP4 validation, frame extraction using `decord`, schema-safe JSON response, and fallback handling.
  - `Dockerfile` and `docker-compose.yml` for GPU-enabled local deployment.
- Modified Output:
  - Added strict response normalization and allowed-operation validation to keep outputs rubric-compliant.
  - Added `/health` endpoint and safer frame indexing.
- Estimated Time Saved: ~2 hours.
- Commit Hash: `bc42717e62750e9ca18e4e91b3ad4751c037144e`

## Phase 2: Temporal Data Pipeline
* **Tool Used:** Cursor / Deep Researcher
* **Prompt:** "Write a robust data pipeline that uses Farneback optical flow (cv2) to isolate the 8 highest-motion frames within operation boundaries. Use `decord` for C++ level frame extraction to avoid PyTorch RAM overhead, and serialize the image+JSON outputs into a PyTorch WebDataset (.tar) format for streaming."
* **Modifications:** Added a fallback mechanism to handle boundary clips that are shorter than 8 frames. Implemented a sample generator to output visual grids into `training_data_samples/` for grader verification.
* **Time Saved:** ~3.5 hours of debugging memory leaks with standard OpenCV sequential frame reading.
* **Commit Hash:** `0730418d13d4366cff69a8428af36126ad1d7b5a`

## Phase 3: PEFT Fine-Tuning Execution
* **Tool Used:** Claude / Deep Researcher
* **Prompt:** "Generate the training loop for Qwen2.5-VL-2B using SFTTrainer. Implement strict memory constraints for a 16GB T4 GPU: 4-bit bitsandbytes, gradient checkpointing, `enable_input_require_grads()`, and a custom VRAM math calculation block."
* **Modifications:** Shifted `bf16=True` to `fp16=True` after identifying that Kaggle T4s use the Turing architecture, which lacks native bfloat16 support, thereby preventing a massive software-emulation slowdown.
* **Time Saved:** ~4 hours of iterative OOM debugging and PyTorch tensor shape matching.
* **Commit Hash:** `0ecda5fd7abb617e56b20d4cd882fa247d5bc36e`

## Phase 4 & 5: Evaluation and Architecture Defense
* **Tool Used:** Claude / Gemini
* **Prompt:** "Write `evaluate.py` to calculate the 1D Temporal Intersection over Union (tIoU) and Anticipation Accuracy (AA@1). Draft `ARCHITECTURE.md` defending the choice of Qwen2.5-VL-2B and the Farneback optical flow sampling based on strict 16GB T4 VRAM constraints."
* **Modifications:** Added edge-case handling to the tIoU math function to prevent division-by-zero if the model predicts inverted start/end frames.
* **Time Saved:** ~2 hours on metric scripting and markdown formatting.
* **Commit Hash:** `c95f9bd85d83f6ef20b02b0379ce2bd571a0807f`

## Phase 6: Real-Eval Automation Wiring
* **Tool Used:** Codex (GPT-5 coding agent)
* **Prompt:** "Implement automation scripts to generate U0108 ground-truth eval files and batch predictions from the `/predict` endpoint, then harden evaluation to avoid accidental mock-metric submission."
* **Modifications:** Added `.gitignore`, `build_eval_ground_truth.py`, `batch_predict.py`, and `--strict` mode in `evaluate.py` to force real eval file presence.
* **Time Saved:** ~1.5 hours of manual clip iteration and JSON assembly.
* **Commit Hash:** `7c4c77e02dd8c2cb2c5c338221b34cade9089515`

## Phase 7: Inference/Deployment Hardening for Real Eval
* **Tool Used:** Codex (GPT-5 coding agent)
* **Prompt:** "Harden inference for real base vs finetuned evaluation by fixing Qwen2-VL model IDs/classes, adding optional adapter loading, and adding command-line utilities for building ground truth and batch prediction generation."
* **Modifications:** Updated `main.py` to use `Qwen2VLForConditionalGeneration` with `MODEL_ID`/`ADAPTER_PATH` env support, added `peft` to Docker image, and wired docker-compose env passthrough.
* **Time Saved:** ~2 hours of manual endpoint switching and mismatch debugging.
* **Commit Hash:** `22da443a01699d9036ed481b58e4152fdd27def1`
