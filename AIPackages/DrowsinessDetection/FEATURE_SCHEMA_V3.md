# Drowsiness feature schema v3

**Schema version:** `drowsiness_feature_schema_v3`  
**Feature count:** `12`  
**Authoritative Python definition:** `feature_contract.py`  
**Authoritative Swift definition:** `DrowsinessFeatureContract` in `DrowsinessFeatureSample.swift`  
**Pure math (Python):** `feature_math.py`  
**On-device extractor (Swift):** `DrowsinessFeatureExtractor.swift`

## Feature names (exact order)

1. `face_detected`
2. `yaw`
3. `pitch`
4. `roll`
5. `left_eye_valid`
6. `right_eye_valid`
7. `left_eye_aspect_ratio`
8. `right_eye_aspect_ratio`
9. `left_pupil_rel_x`
10. `left_pupil_rel_y`
11. `right_pupil_rel_x`
12. `right_pupil_rel_y`

## Coordinate conventions

- Apple Vision **image** coordinates after converting face-normalized landmark points through the face bounding box.
- Origin: **bottom-left** (Vision default).
- `+x` right, `+y` up.

## Eye geometry

For each eye contour `eye_points`:

```text
eye_min_x = min(x)
eye_max_x = max(x)
eye_min_y = min(y)
eye_max_y = max(y)
eye_width  = eye_max_x - eye_min_x
eye_height = eye_max_y - eye_min_y
eps = 1e-6
```

### EAR (bounding-box convention)

```text
EAR = eye_height / max(eye_width, eps)
```

### Pupil-relative coordinates

```text
pupil_rel_x = (pupil_x - eye_min_x) / max(eye_width, eps)
pupil_rel_y = (pupil_y - eye_min_y) / max(eye_height, eps)
```

Clamp valid relative coordinates to `[0.0, 1.0]`.

## Validity

An eye is valid (`eye_valid = 1`) only when:

- Face is detected.
- Eye landmark region exists with at least **4** points.
- All eye points are finite.
- `eye_width > eps` and `eye_height > eps`.
- EAR is finite.
- Pupil exists with finite coordinates.

Otherwise the eye features are zeroed:

```text
eye_valid = 0
eye_aspect_ratio = 0
pupil_rel_x = 0
pupil_rel_y = 0
```

### Missing face

```text
face_detected = 0
all remaining features = 0.0
```

One feature row is still emitted for every decoded video frame (never skip).

## Removed from model input (v2 → v3)

`left_eyelid_gap_ratio`, `right_eyelid_gap_ratio`.

## CSV output

New extractions files use:

```text
*_rgb_face.apple_drowsiness_v3.csv
```

Convert from v2 with:

```bash
python scripts/build_v3_csv_from_legacy.py --force
```

Do not overwrite v1/v2 files.

## Server / client rejection

- Request `feature_schema_version` must equal `drowsiness_feature_schema_v3`.
- `feature_names` must match the list above **exactly** (names and order).
- Checkpoints trained on v1 (22-D) or v2 (14-D) must not be served once v3 is deployed.
