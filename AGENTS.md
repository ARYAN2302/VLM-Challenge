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
- Commit Hash: TBD (to be updated after commit).
