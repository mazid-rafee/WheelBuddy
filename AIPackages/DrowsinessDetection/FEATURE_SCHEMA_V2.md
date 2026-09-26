# Drowsiness feature schema v2

> **Superseded by [`FEATURE_SCHEMA_V3.md`](FEATURE_SCHEMA_V3.md)** (`drowsiness_feature_schema_v3`, 12 features). Kept for historical reference only.

**Schema version:** `drowsiness_feature_schema_v2`  
**Feature count:** `14`  
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
9. `left_eyelid_gap_ratio`
10. `right_eyelid_gap_ratio`
11. `left_pupil_rel_x`
12. `left_pupil_rel_y`
13. `right_pupil_rel_x`
14. `right_pupil_rel_y`

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

### EAR (unchanged bounding-box convention)

```text
EAR = eye_height / max(eye_width, eps)
```

### Eyelid gap ratio

Upper eyelid point = eye landmark with **maximum y**.  
Lower eyelid point = eye landmark with **minimum y**.

```text
eyelid_gap = hypot(upper.x - lower.x, upper.y - lower.y)
eyelid_gap_ratio = eyelid_gap / max(eye_width, eps)
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
- EAR and eyelid gap ratio are finite.
- Pupil exists with finite coordinates.

Otherwise the eye features are zeroed:

```text
eye_valid = 0
eye_aspect_ratio = 0
eyelid_gap_ratio = 0
pupil_rel_x = 0
pupil_rel_y = 0
```

### Missing face

```text
face_detected = 0
all remaining features = 0.0
```

One feature row is still emitted for every decoded video frame (never skip).

## Removed from model input (v1 → v2)

`vision_confidence`, raw `left/right_eyelid_gap`, raw `left/right_pupil_{x,y}`,
and all mouth / hand columns.

## CSV output

New extractions files use:

```text
*_rgb_face.apple_drowsiness_v2.csv
```

Do not overwrite v1 `*.apple_drowsiness.csv` until v2 validation passes.

## Server / client rejection

- Request `schema_version` must equal `drowsiness_feature_schema_v2`.
- `feature_names` must match the list above **exactly** (names and order).
- Checkpoints trained on the v1 22-D schema must not be served once v2 is deployed.
