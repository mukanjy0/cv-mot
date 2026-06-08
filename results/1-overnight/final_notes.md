# Overnight Experiment Notes

Updated: 2026-06-07T08:37:44.168229+00:00

## Selected Runs

- `m1_baseline`: `m1_yolo11_sort_smoke`, MOTA=0.1371895679252627, IDF1=0.33574196466198647, FP=11869, FN=43030, IDS=515, FPS=40.27507750069532
- `m2_best`: `m2_yolo26_bytetrack_smoke`, MOTA=0.16946671856753603, IDF1=0.3409278893149861, FP=8800, FN=44157, IDS=384, FPS=56.54825682652509
- `m3_best`: `m3_rtdetr_botsort_conf055`, MOTA=0.2142312183729077, IDF1=0.4363864229765013, FP=8832, FN=41532, IDS=102, FPS=4.625592264164066
- `m4_sahi_best`: `m4_sahi_m2_slice640_overlap020_strict`, MOTA=0.005543012845465212, IDF1=0.37679975001802757, FP=29780, FN=33421, IDS=668, FPS=3.944010785084033

## SAHI Status

- SAHI completed full validation. Compare it against M2 using MOTA 0.005543012845465212 vs 0.16946671856753603, IDF1 0.37679975001802757 vs 0.3409278893149861, FP 29780 vs 8800, FN 33421 vs 44157, and FPS 3.944010785084033 vs 56.54825682652509.

## Failures

- No failed attempts recorded.

Full command history is in `commands.txt`. Per-attempt stdout, stderr, and combined logs are in `logs/`.
