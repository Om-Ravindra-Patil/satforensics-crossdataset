import os, csv
import numpy as np
import torch
from tqdm import tqdm
import cv2
import tifffile
import rasterio
import importlib.util
from sklearn.metrics import classification_report

# ================= CONFIG ================= #
TRAIN_SCRIPT = "/home/c3068579/Downloads/Train_DINOV3.py"
CKPT_PATH    = "satforensics_dino_film_flow.pth"

DATA_DIR     = "/home/c3068579/Documents/inference_results_vc10/DiffusionSat/outputs_fmow_real_jpg"
CSV_OUT      = "output_calibrated.csv"

SAMPLE_FRAC  = 0.20  
IMG_EXTS     = ('.tif', '.tiff', '.jp2', '.png', '.jpg', '.jpeg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= GROUND TRUTH HELPERS ================= #
def is_ground_truth_manipulated(path):
    """
    Non-SN2 imagery (fMoW datasets) represent the manipulated pool.
    """
    return "SN2_" not in os.path.basename(path)


# ================= LOAD MODULE & IMAGES ================= #
def load_module(path):
    spec = importlib.util.spec_from_file_location("sat", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_image(path, size):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff'):
        try: img = tifffile.imread(path)
        except:
            with rasterio.open(path) as src: img = src.read().transpose(1,2,0)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None: raise ValueError(path)
        if len(img.shape) == 3: img = img[..., ::-1]

    if img.ndim == 2: img = np.stack([img]*3, -1)
    if img.shape[2] > 3: img = img[..., :3]
    img = img.astype(np.float32)
    mx = img.max() if img.size else 1.0
    img = img / (255.0 if mx > 1.5 else 1.0)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


# ================= SCORE ENGINE ================= #
def score_tile(model, T, rgb, radio, modis):
    with torch.no_grad():
        out = model(rgb, radio, modis)
    if not isinstance(out, (tuple, list)): return None

    rec, latent, aux = out[0], out[1] if len(out) > 1 else None, out[2] if len(out) > 2 else None
    rec_err = (rec - rgb).abs().mean().item()
    latent_err = latent.abs().mean().item() if torch.is_tensor(latent) else 0.0
    aux_err = aux.abs().mean().item() if torch.is_tensor(aux) else 0.0

    score = (0.6 * rec_err) + (0.3 * latent_err) + (0.1 * aux_err)
    return rec_err, latent_err, aux_err, score


# ================= MAIN ================= #
def main():
    T = load_module(TRAIN_SCRIPT)
    blob = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model = T.DinoRadModisAE(use_modis=True).to(device)
    model.load_state_dict(blob["state_dict"], strict=False)
    model.eval()
    size = T.ENCODER_SIZE

    # 1. Gather files
    all_imgs = []
    for r, _, fs in os.walk(DATA_DIR):
        for f in fs:
            if f.lower().endswith(IMG_EXTS):
                all_imgs.append(os.path.join(r, f))

    np.random.shuffle(all_imgs)
    selected_imgs = all_imgs[:max(1, int(len(all_imgs) * SAMPLE_FRAC))]

    # Isolate pools by ground truth
    all_genuine = [p for p in selected_imgs if not is_ground_truth_manipulated(p)]
    all_manipulated = [p for p in selected_imgs if is_ground_truth_manipulated(p)]

    np.random.shuffle(all_genuine)
    np.random.shuffle(all_manipulated)

    # 10% of the manipulated pool defines the evaluation block size
    num_manipulated = len(all_manipulated)
    eval_size = max(1, int(num_manipulated * 0.10))

    # --- ALLOCATE SPLITS ---
    # Evaluation block: Exactly balanced (50% genuine / 50% manipulated)
    eval_imgs = all_genuine[:eval_size] + all_manipulated[:eval_size]
    
    # Tuning block: 100% Genuine (used to find FDR/quantile threshold on clean data)
    tune_imgs = all_genuine[eval_size : eval_size + eval_size]
    
    # Calibration block: 100% Genuine reference data
    sn2_ref_imgs = all_genuine[eval_size + eval_size:]
    
    # Remainder of manipulated images are unused
    unused_imgs = all_manipulated[eval_size:]

    print(f"[INFO] Data Routing Summary:")
    print(f"       -> Calibration (sn2_ref): {len(sn2_ref_imgs)} (100% Genuine)")
    print(f"       -> Tune Set (tune_10):   {len(tune_imgs)} (100% Genuine)")
    print(f"       -> Eval Set (eval_10):   {len(eval_imgs)} (Exactly 50% Genuine / 50% Manipulated)")

    # 2. Process and score everything
    rows = []
    for p in tqdm(selected_imgs, desc="Scoring dataset"):
        try:
            img = load_image(p, size)
            rgb = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
            radio = torch.zeros(1, T.RADIO_SPATIAL_GRID, T.RADIO_SPATIAL_GRID, T.RADIO_DIM, device=device)
            modis = torch.tensor([0], device=device)

            rec_err, latent_err, aux_err, score = score_tile(model, T, rgb, radio, modis)
            
            # Determine split allocation mapping
            if p in sn2_ref_imgs: split_label = "sn2_ref"
            elif p in tune_imgs: split_label = "tune_10"
            elif p in eval_imgs: split_label = "eval_10"
            else: split_label = "unused"

            gt_label = "manipulated" if is_ground_truth_manipulated(p) else "genuine"

            rows.append({
                "image": p,
                "split": split_label,
                "rec_err": rec_err,
                "latent_err": latent_err,
                "aux_err": aux_err,
                "score": score,
                "ground_truth": gt_label
            })
        except Exception as e:
            print("[SKIP]", p, e)

    # 3. TUNE STAGE: Since tune_10 is entirely genuine, pick a threshold based on target False Positive Rate
    tune_rows = [r for r in rows if r["split"] == "tune_10"]
    
    if len(tune_rows) > 0:
        tune_scores = [r["score"] for r in tune_rows]
        # Set threshold at the 5th percentile of genuine data (allows a 5% False Positive Rate target)
        best_threshold = np.percentile(tune_scores, 5)
        print(f"\n[TUNING COMPLETE] Threshold set at 5th percentile of genuine data: {best_threshold:.4f}")
    else:
        ref_scores = [r["score"] for r in rows if r["split"] == "sn2_ref"]
        best_threshold = np.percentile(ref_scores, 5) if ref_scores else 0.0
        print(f"\n[TUNING FALLBACK] Using 5th percentile calibration baseline: {best_threshold:.4f}")

    # 4. EVALUATE STAGE: Assign final labels using the tuned threshold
    for r in rows:
        r["threshold"] = best_threshold
        if r["split"] == "sn2_ref":
            r["label"] = "genuine"  
        else:
            r["label"] = "manipulated" if r["score"] < best_threshold else "genuine"

    # Print evaluation metrics for the balanced test block
    eval_rows = [r for r in rows if r["split"] == "eval_10"]
    if eval_rows:
        y_true = [r["ground_truth"] for r in eval_rows]
        y_pred = [r["label"] for r in eval_rows]
        
        eval_correct = sum(1 for r in eval_rows if r["label"] == r["ground_truth"])
        print(f"\n[EVALUATION COMPLETE] Isolated Test Split Accuracy: {(eval_correct / len(eval_rows)) * 100:.2f}%")
        
        print("\n" + "="*30 + " CLASSIFICATION REPORT (BALANCED EVAL SPLIT) " + "="*30)
        print(classification_report(y_true, y_pred, digits=4))
        print("="*96 + "\n")

    # 5. Export results to output_calibrated.csv
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print("DONE ->", CSV_OUT)


if __name__ == "__main__":
    main()
