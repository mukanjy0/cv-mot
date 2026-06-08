# Next Experiments Report

Updated: 2026-06-07T20:48:36.455098+00:00

## Experiment List

- Detection-only diagnostics: configs/overnight/m2_yolo26_bytetrack_conf035.yaml, configs/overnight/m3_rtdetr_botsort_conf055.yaml
- M3 confidence sweep: 0.50, 0.55, 0.60, 0.65.
- Upscaling: YOLO26 1.5x/2.0 with ByteTrack and BoT-SORT, plus RT-DETR 1.5x BoT-SORT.
- Strict SAHI + BoT-SORT: slice 640/768, overlap 0.20/0.15, conf 0.40/0.50.
- GMC ablation: RT-DETR and YOLO26 BoT-SORT with `cmc_method: sof` versus disabled.

## Best Methods

- Best by MOTA: `rtdetr_botsort_conf055_gmc_on` with MOTA=0.2142312183729077, MOTA=0.2142312183729077, IDF1=0.4363864229765013, FP=8832, FN=41532, IDS=102, FPS=4.1975867571186.
- Best by IDF1: `m3_rtdetr_botsort_conf050` with IDF1=0.46515581296514097, MOTA=0.21074347995328924, IDF1=0.46515581296514097, FP=11643, FN=38914, IDS=133, FPS=4.018700649165206.
- Best by FPS: `yolo26_botsort_gmc_off` with FPS=53.96834416667575, MOTA=0.17111716621253403, IDF1=0.33244176211050913, FP=5974, FN=46938, IDS=323, FPS=53.96834416667575.
- Best on uav0000305_00000_v: `m3_rtdetr_botsort_conf050` with MOTA=0.26070226070226066, MOTA=0.26070226070226066, IDF1=0.4321447578499202, FP=189, FN=2868, IDS=17, FPS=3.981472728546189.

## uav0000305_00000_v Detection Diagnosis

- `m2_yolo26_bytetrack_conf035` class `pedestrian`: precision=0.0, recall=0.0, AP50=0.0, FP=0, FN=540, GT=540, pred=0.
- `m2_yolo26_bytetrack_conf035` class `car`: precision=0.8140845070422535, recall=0.1597567716970702, AP50=0.1414265720333477, FP=132, FN=3040, GT=3618, pred=710.
- `m2_yolo26_bytetrack_conf035` class `all`: precision=0.8140845070422535, recall=0.139009139009139, AP50=0.12305948475628957, FP=132, FN=3580, GT=4158, pred=710.
- `m3_rtdetr_botsort_conf055` class `pedestrian`: precision=0.0, recall=0.0, AP50=0.0, FP=8, FN=540, GT=540, pred=8.
- `m3_rtdetr_botsort_conf055` class `car`: precision=0.889584964761159, recall=0.31398562741846325, AP50=0.2982925854334514, FP=141, FN=2482, GT=3618, pred=1277.
- `m3_rtdetr_botsort_conf055` class `all`: precision=0.8840466926070039, recall=0.27320827320827323, AP50=0.2592335179156308, FP=149, FN=3022, GT=4158, pred=1285.

## Interpretation

- Upscaling car-recall comparison: best upscaled method `yolo26_upscale20_botsort` has proxy=0.4987 versus baseline=0.2989 (delta=+0.1997).
- SAHI comparison: best strict SAHI car proxy is `sahi_yolo26_botsort_slice768_overlap015_conf040` at 0.4727 versus baseline=0.2989; aggregate FP=22507.0.
- GMC comparison: rtdetr_botsort_conf055: IDS off-on delta=192.0, MOTA on-off delta=0.008127676138575235; yolo26_botsort: IDS off-on delta=147.0, MOTA on-off delta=0.008470221876216488.

Detector-limited failures are indicated by high FN and low detection recall.
Association-limited failures are indicated by high IDS, many unique IDs, and many tracks of three frames or fewer.
Precision failures are indicated by high FP, especially when SAHI increases predicted boxes without improving recall.
Speed tradeoffs are indicated by low FPS; compare FPS before choosing an upscaled or sliced method.

## Recommendation

Recommended final method: `rtdetr_botsort_conf055_gmc_on` for the current selection rule of highest full-validation MOTA, with IDF1/FPS checked for presentation tradeoffs.

Presentation narrative: start with uav0000305_00000_v as the visible failure case, separate missed cars from ID fragmentation, then show the upscaling, SAHI, and GMC ablations as targeted tests rather than blind hyperparameter tuning.
