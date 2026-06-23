import os, glob, numpy as np

INFER_DIR = '/home/c3068579/Documents/inference_results_vc10/DiffusionSat/outputs_fmow_real/'
SAVE_HEATMAPS = True
HEATMAP_DIR = 'infer_heatmaps'
if SAVE_HEATMAPS: os.makedirs(HEATMAP_DIR, exist_ok=True)

paths = []
for ext in ('*.tif','*.tiff','*.jp2','*.TIF','*.TIFF','*.JP2'):
    paths += glob.glob(os.path.join(INFER_DIR, '**', ext), recursive=True)
paths = sorted(p for p in paths if not os.path.splitext(p)[0].lower().endswith('_mask'))
print(f"{len(paths)} images to score\n")

rows, flagged = [], 0
for p in paths:
    hm = os.path.join(HEATMAP_DIR, os.path.splitext(os.path.basename(p))[0] + '.png') if SAVE_HEATMAPS else None
    try:
        r = classify_image(p, model, calib, device, heatmap_path=hm)
    except Exception as e:
        print(f"  FAILED {os.path.basename(p)}: {e}"); continue
    rows.append(r); flagged += int(r['label'] == 'manipulated')
    print(f"{r['label']:11s} score={r['image_score']:.3f} thr={r['threshold']:.3f} "
          f"margin={r['margin']:+.3f}  {os.path.basename(p)}")

if rows:
    sc = np.array([r['image_score'] for r in rows])
    mg = np.array([r['margin'] for r in rows])
    print(f"\n=== {flagged}/{len(rows)} flagged manipulated "
          f"({100*flagged/len(rows):.1f}% detection rate) ===")
    print(f"score: min={sc.min():.3f} med={np.median(sc):.3f} max={sc.max():.3f} | "
          f"threshold={rows[0]['threshold']:.3f}")
    print(f"margin>0 means above threshold; median margin={np.median(mg):+.3f}")
