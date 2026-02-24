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
