import os, csv, importlib.util
import numpy as np
import torch
from tqdm import tqdm

# ============================ CONFIG ============================ #
TRAIN_SCRIPT    = "/home/c3068579/Downloads/Train_DINOV3.py"
CKPT_PATH       = "satforensics_dino_film_flow.pth"

DATA_DIR        = "/home/c3068579/Documents/inference_results_vc10/DiffusionSat/outputs_fmow_real_5pct_jpg"
REAL_PREFIX     = "sn2"

MODIS_PATH      = "/home/c3068579/Documents/wholeworld_epsg4326.tif"
RADIO_CACHE_DIR = "/home/c3068579/Documents/radio_cache_dino"

CSV_OUT         = "inference_results.csv"

USE_MODIS       = True
# =============================================================== #

IMG_EXTS = ('.tif', '.tiff', '.jp2', '.png', '.jpg', '.jpeg')


# ---------------- NORMALISATION ---------------- #
def _to_float01(img):
    img = np.asarray(img).astype(np.float32)
    mx = float(img.max()) if img.size else 1.0

    if   mx <= 1.5:     denom = 1.0
    elif mx <= 255.0:   denom = 255.0
    elif mx <= 4095.0:  denom = 4095.0
    elif mx <= 65535.0: denom = 65535.0
    else:               denom = mx

    return np.clip(img / denom, 0.0, 1.0)


# ---------------- LOAD TRAIN MODULE ---------------- #
def load_train_module(path, modis_path):
    import rasterio

    spec = importlib.util.spec_from_file_location("satforensics_train", path)
    mod = importlib.util.module_from_spec(spec)

    orig_open = rasterio.open
    first = {"used": False}

    def _patched(p, *a, **k):
        if first["used"]:
            return orig_open(p, *a, **k)
        first["used"] = True
        return orig_open(modis_path, *a, **k)

    rasterio.open = _patched
    try:
        spec.loader.exec_module(mod)
    finally:
        rasterio.open = orig_open

    mod.MODIS_PATH = modis_path
    return mod


def configure_paths(T):
    import rasterio
    T.RADIO_CACHE_DIR = RADIO_CACHE_DIR
    T.USE_MODIS = USE_MODIS
    T.modis_src = rasterio.open(MODIS_PATH) if USE_MODIS else None


# ---------------- IMAGE LOADING ---------------- #
def find_images(root):
    out = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                out.append(os.path.join(r, f))
    return sorted(out)


def load_image(path):
    import tifffile, rasterio, cv2

    ext = os.path.splitext(path)[1].lower()

    try:
        if ext in ('.tif', '.tiff'):
            try:
                img = tifffile.imread(path)
            except:
                with rasterio.open(path) as src:
                    img = src.read().transpose(1,2,0)
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("cv2 failed")

            if img.ndim == 3:
                img = img[..., ::-1]

        img = np.asarray(img)

        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)

        if img.shape[2] > 3:
            img = img[..., :3]

        return _to_float01(img)

    except Exception as e:
        raise RuntimeError(f"Failed loading {path}: {e}")


# ---------------- MAIN ---------------- #
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T = load_train_module(TRAIN_SCRIPT, MODIS_PATH)
    configure_paths(T)

    blob = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    calib = blob.get("calibration", {})
    state = blob["state_dict"]

    model = T.DinoRadModisAE(
        use_modis=calib.get("use_modis", USE_MODIS)
    ).to(device)

    model.load_state_dict(state, strict=False)
    model.eval()

    imgs = find_images(DATA_DIR)

    print(f"[ok] total images: {len(imgs)}")

    rows = []

    threshold = calib.get("threshold", 0.5)

    for p in tqdm(imgs, desc="infer"):

        try:
            img = load_image(p)
        except Exception as e:
            print("[skip]", p, e)
            continue

        H, W = img.shape[:2]

        ys = T.compute_adaptive_positions(H, T.ENCODER_SIZE)
        xs = T.compute_adaptive_positions(W, T.ENCODER_SIZE)

        scores = []

        for y in ys:
            for x in xs:

                patch = img[y:y+T.ENCODER_SIZE, x:x+T.ENCODER_SIZE]

                if patch.shape[0] < 16 or patch.shape[1] < 16:
                    continue

                import cv2
                patch = cv2.resize(patch, (T.ENCODER_SIZE, T.ENCODER_SIZE))

                rgb = T._norm_rgb(patch)

                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.detach().cpu().numpy()

                rgb = torch.from_numpy(rgb).unsqueeze(0).to(device)

                modis = torch.tensor([0], device=device)

                radio = torch.zeros(
                    1,
                    T.RADIO_SPATIAL_GRID,
                    T.RADIO_SPATIAL_GRID,
                    T.RADIO_DIM,
                    device=device
                )

                with torch.no_grad():
                    rec, latent, aux = model(rgb, radio, modis)

                err = (rec - rgb).abs().mean().item()
                scores.append(err)

        if not scores:
            continue

        image_score = float(np.mean(sorted(scores)[-max(1, len(scores)//10):]))
        label = "manipulated" if image_score > threshold else "genuine"

        # ===================== NOTIFICATION ===================== #
        if label == "genuine":
            print(f"\n[PRISTINE DETECTED] {p}")
            print(f"score={image_score:.6f} | threshold={threshold}")
        # ======================================================= #

        rows.append({
            "image": p,
            "score": image_score,
            "label": label,
            "threshold": threshold
        })

    if rows:
        with open(CSV_OUT, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print("\nDONE ->", CSV_OUT)


if __name__ == "__main__":
    main()
