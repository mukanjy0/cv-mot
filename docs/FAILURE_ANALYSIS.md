# Failure Analysis

The most useful outcome of this project is not a single score. It is the
separation of MOT failures into detector-limited, association-limited,
precision-limited, and speed-limited cases.

## Failure Taxonomy

| Failure type | Signal | Typical cause | Project evidence |
|---|---|---|---|
| Detector-limited | High FN, low detection recall | small/blurred objects, viewpoint, detector threshold | `uav0000305_00000_v` car recall and pedestrian recall |
| Association-limited | High IDS, many short tracks | camera motion, occlusion, weak appearance cues | GMC ablation and per-sequence diagnostics |
| Precision-limited | High FP | permissive detector threshold, slicing artifacts | SAHI/upscaling tradeoffs |
| Speed-limited | Low FPS | large detector, slicing, upscaling | RT-DETR and SAHI runtime rows |

## Highlighted Crossroad Sequence

Sequence: `uav0000305_00000_v`

This aerial crossroad sequence contains many small cars and pedestrians. The
detection-only evaluation shows that tracking is bottlenecked by missed
detections:

| Detector | Class | Precision | Recall | AP50 | FN |
|---|---|---:|---:|---:|---:|
| YOLO26 | car | 0.814 | 0.160 | 0.141 | 3040 |
| RT-DETR conf055 | car | 0.890 | 0.314 | 0.298 | 2482 |
| YOLO26 | pedestrian | 0.000 | 0.000 | 0.000 | 540 |
| RT-DETR conf055 | pedestrian | 0.000 | 0.000 | 0.000 | 540 |

RT-DETR nearly doubles car recall versus YOLO26, but most cars are still never
detected. The tracker cannot recover objects that never enter the detection
stream.

Visual evidence:

- `slides/main.pdf`, crossroad visual slides;
- `slides/visual_revision_notes.md`, crossroad asset notes.

## GMC and Camera Motion

GMC was tested by turning BoT-SORT camera-motion compensation on and off while
holding the detector family fixed.

| Family | MOTA off | MOTA on | IDF1 off | IDF1 on | IDS off | IDS on |
|---|---:|---:|---:|---:|---:|---:|
| RT-DETR + BoT-SORT conf055 | 0.206 | 0.214 | 0.391 | 0.436 | 294 | 102 |
| YOLO26 + BoT-SORT | 0.171 | 0.180 | 0.332 | 0.370 | 323 | 176 |

The main effect is identity stability, not a new detector. GMC reduces RT-DETR
identity switches by 192 and YOLO26 identity switches by 147 in aggregate.

Visual evidence:

- `slides/main.pdf`, GMC off/on slides;
- `slides/visual_revision_notes.md`, GMC asset notes.

`uav0000182_00000_v` is a noted sequence-level exception. The aggregate
conclusion is therefore stated carefully: GMC helped overall, but it is not
guaranteed to help every sequence.

## SAHI and Upscaling

Small-object modes were tested because drone videos contain many tiny targets.

| Method | Car recall proxy | MOTA | IDF1 | FP | FN | IDS |
|---|---:|---:|---:|---:|---:|---:|
| YOLO26 baseline | 0.299 | 0.169 | 0.325 | 6.0k | 47.0k | 332 |
| YOLO26 upscale 2.0 + BoT-SORT | 0.499 | 0.110 | 0.463 | 23.8k | 33.1k | 295 |
| SAHI 768/0.15/conf0.40 | 0.473 | 0.092 | 0.423 | 22.5k | 35.5k | 281 |
| RT-DETR conf055 + GMC | 0.294 | 0.214 | 0.436 | 8.8k | 41.5k | 102 |

Upscaling and SAHI reduce FN and improve the car recall proxy, but they produce
many more FP and unstable tracks. This is the core small-object MOT lesson:
recall gains must be filtered and associated robustly before they improve MOTA.

Visual evidence:

- `slides/main.pdf`, SAHI and upscaling visual slides;
- `slides/visual_revision_notes.md`, SAHI/upscaling asset notes.

## Final Selection

Selected method: `rtdetr_botsort_conf055_gmc_on`.

Why:

- best full-validation MOTA among completed next-step experiments;
- low IDS compared with other strong methods;
- strong IDF1;
- explicitly addresses camera motion through GMC.

Tradeoff:

- slower than YOLO26 methods;
- still detector-limited on the highlighted crossroad sequence.

## Recommended Next Experiments

- Fine-tune or adapt the detector on VisDrone classes and scale distribution.
- Add stricter postprocessing for SAHI/upscaling to reduce false tracks.
- Explore sequence-adaptive thresholds for crossroad versus lower-density
  scenes.
- Evaluate stronger ReID embeddings or tracker settings for occlusion-heavy
  sequences.
