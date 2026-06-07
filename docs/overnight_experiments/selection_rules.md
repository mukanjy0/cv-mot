# Overnight Selection Rules

Selections are made twice for tuning stages:

1. Rank compact screening results and promote only the configured top count.
2. Rank promoted full-validation results and store the winner in
   `stages/<stage>/selection.json`.

The baseline config is included in each M2/M3 grid, so a tuning candidate
cannot displace it unless its ranking is better.

## M3: RT-DETR + BoT-SORT

Lexicographic order:

1. Highest MOTA.
2. Highest IDF1.
3. Lowest FP.
4. Highest FPS.

MOTA is primary because the current M3 failure is excessive FP. Report FP
reduction relative to the M3 baseline alongside MOTA and IDF1. A stricter
config that reduces FP but loses enough recall to lower MOTA does not win.

If no promoted M3 config finishes all seven sequences, use
`configs/m3_rtdetr_botsort_smoke.yaml`.

## M2: YOLO26 + ByteTrack

Lexicographic order:

1. Highest MOTA.
2. Highest IDF1.
3. Highest FPS.
4. Lowest FP.

M2 is the speed-oriented method, so FPS is the third tie-breaker. Selection is
still based on full-validation accuracy first.

If no promoted M2 config finishes all seven sequences, use
`configs/m2_yolo26_bytetrack_smoke.yaml`.

## M4: SAHI + YOLO26 + ByteTrack

Among successful SAHI candidates:

1. Highest MOTA.
2. Highest IDF1.
3. Lowest FN.
4. Lowest FP.
5. Highest FPS.

The best SAHI candidate is included in the final table whenever it completes
all seven sequences. Interpret it against `m2_best`, not in isolation:

- FN delta: whether slicing finds more small/distant objects.
- FP delta: whether slice merging adds duplicate or background detections.
- MOTA and IDF1 delta: whether detection gains improve tracking quality.
- FPS ratio: the actual runtime cost.

For the written recommendation, prefer M4 over M2 only if it improves MOTA, or
if MOTA is within 0.01 while IDF1/FN improve enough to justify the measured
speed loss. The runner records results but does not hide a slower or worse M4.

If SAHI does not complete, omit M4 from the final table and document the error
in `summaries/final_notes.md`.
