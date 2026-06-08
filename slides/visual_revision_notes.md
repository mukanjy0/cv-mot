# Visual Revision Notes

## Summary

Revised `slides/main.tex` from 15 slides to 22 slides by replacing crowded contact-sheet slides with large still-frame visual comparisons. No videos are embedded.

## Generated Visual Assets

All revised visual panels are under `slides/img/visuals/`. They were rendered from raw VisDrone frames plus existing `tracks.txt` outputs. Raw frames came from `VisDrone2019-MOT-val/sequences/`.

### Crossroad sequence: `uav0000305_00000_v`

Frame 58 is used for the main detector-limited failure visuals because the manifest identified it as a GT-heavy crossroad frame.

- `crossroad_gt_vs_yolo26_frame058_yolo26.jpg`
  - Source tracks: `outputs/overnight/20260607T073428Z/stages/tune_m2/promoted/m2_yolo26_bytetrack_smoke/attempts/001/benchmark/runs/m2_yolo26_bytetrack_smoke/uav0000305_00000_v/tracks.txt`
  - Used on slide 8.
- `crossroad_gt_vs_yolo26_frame058_gt.jpg`
  - Source annotations: `VisDrone2019-MOT-val/annotations/uav0000305_00000_v.txt`
  - Used on slide 8.
- `crossroad_gt_vs_rtdetr_frame058_rtdetr055.jpg`
  - Source tracks: `outputs/next_experiments/default/stages/gmc_ablation/rtdetr_botsort_conf055_gmc_on/attempts/001/benchmark/runs/rtdetr_botsort_conf055_gmc_on/uav0000305_00000_v/tracks.txt`
  - Used on slide 9.
- `crossroad_gt_vs_rtdetr_frame058_gt.jpg`
  - Source annotations: `VisDrone2019-MOT-val/annotations/uav0000305_00000_v.txt`
  - Used on slide 9.
- `crossroad_yolo26_vs_rtdetr_frame058_yolo26.jpg`
  - Source tracks: YOLO26 promoted baseline listed above.
  - Used on slide 16.
- `crossroad_yolo26_vs_rtdetr_frame058_rtdetr055.jpg`
  - Source tracks: RT-DETR conf055 GMC-on listed above.
  - Used on slide 16.

Frame 22 is used for the crossroad conf050 and SAHI comparisons because the completed tracks show clearer differences for those variants there.

- `crossroad_rtdetr055_vs_rtdetr050_frame022_rtdetr055.jpg`
  - Source tracks: RT-DETR conf055 GMC-on listed above.
  - Used on slide 11.
- `crossroad_rtdetr055_vs_rtdetr050_frame022_rtdetr050.jpg`
  - Source tracks: `outputs/next_experiments/default/stages/m3_compact_tuning/m3_rtdetr_botsort_conf050/attempts/001/benchmark/runs/m3_rtdetr_botsort_conf050/uav0000305_00000_v/tracks.txt`
  - Used on slide 11.
- `crossroad_rtdetr055_vs_sahi_frame022_rtdetr055.jpg`
  - Source tracks: RT-DETR conf055 GMC-on listed above.
  - Used on slide 17.
- `crossroad_rtdetr055_vs_sahi_frame022_sahi768.jpg`
  - Source tracks: `outputs/next_experiments/default/stages/strict_sahi_botsort/sahi_yolo26_botsort_slice768_overlap015_conf040/attempts/001/benchmark/runs/sahi_yolo26_botsort_slice768_overlap015_conf040/uav0000305_00000_v/tracks.txt`
  - Used on slide 17.

Frame 58 variants for RT-DETR conf050 and SAHI were also generated but are not used in the deck because frame 22 makes those comparisons clearer.

### GMC sequence: `uav0000339_00001_v`

This is the selected GMC visual sequence from the manifest, excluding `uav0000182_00000_v`.

- `gmc_off_vs_on_frame036_off.jpg`
  - Source tracks: `outputs/next_experiments/default/stages/gmc_ablation/rtdetr_botsort_conf055_gmc_off/attempts/001/benchmark/runs/rtdetr_botsort_conf055_gmc_off/uav0000339_00001_v/tracks.txt`
  - Used on slide 13.
- `gmc_off_vs_on_frame036_on.jpg`
  - Source tracks: `outputs/next_experiments/default/stages/gmc_ablation/rtdetr_botsort_conf055_gmc_on/attempts/001/benchmark/runs/rtdetr_botsort_conf055_gmc_on/uav0000339_00001_v/tracks.txt`
  - Used on slide 13.
- `gmc_off_vs_on_frame146_off.jpg`
  - Source tracks: GMC-off tracks listed above.
  - Used on slide 14.
- `gmc_off_vs_on_frame146_on.jpg`
  - Source tracks: GMC-on tracks listed above.
  - Used on slide 14.

### SAHI/upscaling sequence: `uav0000137_00458_v`

Frame 63 is used because the manifest identified this sequence as a high-FP SAHI tradeoff example.

- `sahi_tradeoff_frame063_rtdetr055.jpg`
  - Source tracks: `outputs/next_experiments/default/stages/gmc_ablation/rtdetr_botsort_conf055_gmc_on/attempts/001/benchmark/runs/rtdetr_botsort_conf055_gmc_on/uav0000137_00458_v/tracks.txt`
  - Used on slide 19.
- `sahi_tradeoff_frame063_sahi768.jpg`
  - Source tracks: `outputs/next_experiments/default/stages/strict_sahi_botsort/sahi_yolo26_botsort_slice768_overlap015_conf040/attempts/001/benchmark/runs/sahi_yolo26_botsort_slice768_overlap015_conf040/uav0000137_00458_v/tracks.txt`
  - Used on slide 19.
- `upscaling_tradeoff_frame063_yolo26.jpg`
  - Source tracks: `outputs/next_experiments/default/stages/upscaling_experiments/m2_yolo26_bytetrack_conf035/attempts/001/benchmark/runs/m2_yolo26_bytetrack_conf035/uav0000137_00458_v/tracks.txt`
  - Used on slide 20.
- `upscaling_tradeoff_frame063_upscale20.jpg`
  - Source tracks: `outputs/next_experiments/default/stages/upscaling_experiments/yolo26_upscale20_botsort/attempts/001/benchmark/runs/yolo26_upscale20_botsort/uav0000137_00458_v/tracks.txt`
  - Used on slide 20.

### Final method success

- `final_success_frame146_rtdetr055_gmc_on.jpg`
  - Sequence/frame: `uav0000339_00001_v`, frame 146.
  - Source tracks: RT-DETR conf055 GMC-on tracks listed above.
  - Used on slide 21.

## Optional / Skippable Slides

The visual slides are designed to be fast and skippable:

- Slide 9: RT-DETR final vs GT on crossroad. Keep if the audience needs the improvement visual after the YOLO26 failure.
- Slide 11: conf055 vs conf050 on crossroad. Keep if explaining the global-vs-sequence tradeoff.
- Slide 14: second GMC frame. Skip if slide 13 makes the point.
- Slide 17: crossroad final vs SAHI. Useful, but visually weaker than the uav0000137 SAHI slide.
- Slide 20: upscaling visual. Optional because slide 18 already carries the metric tradeoff.
- Slide 21: final success visual. Good closing evidence, but safe to skip for time.

## Weak / Caution Slides

- Slide 17 is the weakest visual. On `uav0000305_00000_v`, strict SAHI 768/conf040 adds tracks on frame 22, but it is not as visually cluttered as the high-FP `uav0000137_00458_v` case.
- Slide 20 is illustrative rather than decisive. It supports the recall-oriented upscaling story, but the aggregate metrics are the stronger evidence.

## Recommended 12-15 Slide Fast Subset

For a short presentation, use this subset:

1. Title
2. Problem and Dataset
3. Modular Pipeline
4. Methods Compared
5. YOLO26 vs RT-DETR
6. Tracking: Geometry vs Appearance
7. Baseline Run: First Diagnosis
8. Crossroad Failure: YOLO26 vs Ground Truth
10. RT-DETR Confidence Tuning
12. GMC Ablation: Camera Motion
13. Camera Motion: GMC Off vs On
15. Detection Diagnosis: `uav0000305_00000_v`
16. Crossroad: Detector Choice Matters
18. SAHI and Upscaling: Small-Object Tradeoff
22. Final Conclusion

If tighter, drop slides 4, 6, and 16 only after making sure the project-statement requirements are still covered elsewhere.
