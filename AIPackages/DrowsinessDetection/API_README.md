# DrowsinessDetection Inference API

Local FastAPI server for the DMD eyes-state TCN.

## Model input contract (schema v3)

| Item | Value | Source |
| --- | --- | --- |
| Schema version | `drowsiness_feature_schema_v3` | `feature_contract.py` |
| Architecture | Causal TCN (`GazeZoneTCN` / `arch=tcn`) | `model/model.py` |
| Input tensor | `[batch, T, F]` float32 | `GazeZoneTCN.forward` |
| Window frames `T` | from checkpoint (`window_size`) | checkpoint |
| Feature count `F` | **12** | `DROWSINESS_FEATURE_NAMES` |
| Normalization | Eye-local ratios (see `FEATURE_SCHEMA_V3.md`) | extractor / `feature_math.py` |
| Missing / invalid | zeros + validity flags | contract |
| Live API values | Must be finite; NaN/Inf rejected | API contract |
| Output | Softmax over 3 classes | `torch.softmax(logits)` |

Authoritative docs: [`FEATURE_SCHEMA_V3.md`](FEATURE_SCHEMA_V3.md).

### Exact ordered feature names

```text
face_detected
yaw
pitch
roll
left_eye_valid
right_eye_valid
left_eye_aspect_ratio
right_eye_aspect_ratio
left_pupil_rel_x
left_pupil_rel_y
right_pupil_rel_x
right_pupil_rel_y
```

Legacy schemas (`schema_version=1` / `drowsiness_feature_schema_v2`, including eyelid-gap ratios) are **rejected**.

### Exact class mapping

```text
close      -> closed     -> 0
closing    -> closed     -> 0
open       -> open       -> 1
undefined  -> undefined  -> 2
opening    -> (excluded as window endpoint)
```

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `DROWSINESS_CHECKPOINT_PATH` | `saved_weights/best_loss_v3.pt` | Must be a **v3** checkpoint |
| `DROWSINESS_API_KEY` | unset | When set, require `X-API-Key` |
| `DROWSINESS_DEVICE` | `auto` | `auto` → CUDA if available else CPU |
| `DROWSINESS_HOST` | `0.0.0.0` | Bind host |
| `DROWSINESS_PORT` | `8001` | Bind port |
| `DROWSINESS_SAMPLING_RATE_HZ` | unset | Optional exact rate enforcement |

## Local setup

```bash
cd AIPackages/DrowsinessDetection
python3 -m venv .venv
source .venv/bin/activate
pip install -r api/requirements.txt

# After training a v3 model:
export DROWSINESS_CHECKPOINT_PATH=saved_weights/best_loss_v3.pt
export DROWSINESS_API_KEY=...
export DROWSINESS_SAMPLING_RATE_HZ=15.0

python -m uvicorn api.app:app --host 0.0.0.0 --port 8001
```

## Example request

```json
{
  "feature_schema_version": "drowsiness_feature_schema_v3",
  "session_id": "test-session-1",
  "sequence_id": 42,
  "sent_at_utc": "2026-09-22T18:00:00Z",
  "sampling_rate_hz": 15.0,
  "feature_names": [
    "face_detected", "yaw", "pitch", "roll",
    "left_eye_valid", "right_eye_valid",
    "left_eye_aspect_ratio", "right_eye_aspect_ratio",
    "left_pupil_rel_x", "left_pupil_rel_y",
    "right_pupil_rel_x", "right_pupil_rel_y"
  ],
  "samples": [
    {"timestamp_ms": 1790093823000, "values": [1,0,0,0,1,1,0.25,0.25,0.5,0.5,0.5,0.5]}
  ]
}
```

## CSV conversion

Existing `*_v2.csv` files still contain eyelid-gap-ratio columns. Convert to v3 with:

```bash
python scripts/build_v3_csv_from_legacy.py --force
```

Output:

```text
*_rgb_face.apple_drowsiness_v3.csv
```

Do not overwrite v1/v2 files.
