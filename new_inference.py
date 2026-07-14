import os, csv, warnings
import numpy as np
import torch
from tqdm import tqdm
import cv2
import tifffile
import rasterio
import importlib.util
from scipy import stats
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score, balanced_accuracy_score)
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")

# ================= CONFIG ================= #
TRAIN_SCRIPT = "/home/c3068579/Downloads/Train_DINOV3.py"
CKPT_PATH    = "satforensics_dino_film_flow.pth"

DATA_DIR     = "/home/c3068579/Documents/inference_results_vc10/DiffusionSat/outputs_fmow_real"
CSV_OUT      = "output_johnson_gmm_ablation.csv"

SAMPLE_FRAC  = 1
IMG_EXTS     = ('.tif', '.tiff', '.jp2', '.png', '.jpg', '.jpeg')

SEED         = 42
TARGET_FPR   = 0.05
GMM_KS       = (2, 3)        # mixture sizes to consider
FORCE_MODEL  = "j_gmm_3"     # set to None to re-enable label-free model selection
EPS          = 1e-9

CHANNEL_NAMES = ("rec_err", "latent_err", "aux_err")

# ---------------- ABLATION RUNS ---------------- #
FEATURE_SETS = {
    "rec_only":     [0],
    "latent_only":  [1],
    "aux_only":     [2],
    "rec_latent":   [0, 1],
    "rec_aux":      [0, 2],
    "latent_aux":   [1, 2],
    "full":         [0, 1, 2],
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= GROUND TRUTH ================= #
def is_ground_truth_manipulated(path):
    """Non-SN2 imagery (fMoW datasets) represent the manipulated pool."""
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
        except Exception:
            with rasterio.open(path) as src: img = src.read().transpose(1, 2, 0)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None: raise ValueError(path)
        if len(img.shape) == 3: img = img[..., ::-1]
    if img.ndim == 2: img = np.stack([img] * 3, -1)
    if img.shape[2] > 3: img = img[..., :3]
    img = img.astype(np.float32)
    mx = img.max() if img.size else 1.0
    img = img / (255.0 if mx > 1.5 else 1.0)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


# ================= SCORE ENGINE ================= #
def extract_errors(model, rgb, radio, modis):
    with torch.no_grad():
        out = model(rgb, radio, modis)
    if not isinstance(out, (tuple, list)): return None
    rec = out[0]
    latent = out[1] if len(out) > 1 else None
    aux = out[2] if len(out) > 2 else None
    rec_err = (rec - rgb).abs().mean().item()
    latent_err = latent.abs().mean().item() if torch.is_tensor(latent) else np.nan
    aux_err = aux.abs().mean().item() if torch.is_tensor(aux) else np.nan
    return rec_err, latent_err, aux_err


# ================= NONLINEAR DENSITY SCORER ================= #
# Motivated by the diagnostics on your results CSV:
#   - genuine margins are skewed / heavy-tailed (latent skew +2.2, kurt +7.5)
#     -> Johnson SU marginal Gaussianization (smooth, UNBOUNDED tails, unlike
#        rank/ECDF transforms which clip out-of-support evidence)
#   - the genuine population is multimodal (held-out likelihood strongly
#     prefers a mixture) -> GMM joint density in Gaussianized space
#   - fakes break the genuine rec~aux correlation (-0.73 -> +0.15) and are
#     under-dispersed; a full-covariance mixture density captures both.
# Model selection between {single Gaussian, Johnson-Gaussian, Johnson-quad,
# Johnson-GMM(k)} is by mean log-likelihood on the GENUINE TUNE SPLIT ONLY:
# "which density best describes genuine data" - no manipulated label is ever
# used, so the pipeline remains fully unsupervised.

class _GaussScorer:
    def __init__(self, Z):
        Z = np.atleast_2d(np.asarray(Z, np.float64))
        self.mu = Z.mean(0)
        cov = LedoitWolf().fit(Z - self.mu).covariance_ + EPS * np.eye(Z.shape[1])
        self.cov_inv = np.linalg.pinv(cov)
        _, logdet = np.linalg.slogdet(cov)
        self._c = -0.5 * (Z.shape[1] * np.log(2 * np.pi) + logdet)
    def maha(self, Z):
        D = np.atleast_2d(Z) - self.mu
        return np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', D, self.cov_inv, D), 0.0))
    def log_lik(self, Z):
        D = np.atleast_2d(Z) - self.mu
        return self._c - 0.5 * np.einsum('ij,jk,ik->i', D, self.cov_inv, D)

def _johnson_gaussianize(x_cal):
    """
    ML-fit Johnson SU to the genuine calibration margin; return a smooth,
    strictly monotone map to ~N(0,1) with unbounded tails, so evidence far
    outside the calibration support keeps its magnitude (the ECDF/copula
    alternative clips it - catastrophic for the inverted rec channel).
    Falls back to a robust-z of log1p if the ML fit degenerates.
    """
    x_cal = np.asarray(x_cal, np.float64)
    try:
        prm = stats.johnsonsu.fit(x_cal)
        def f(x):
            u = np.clip(stats.johnsonsu.cdf(np.asarray(x, np.float64), *prm),
                        1e-12, 1 - 1e-12)
            return stats.norm.ppf(u)
        z = f(x_cal)
        if np.isfinite(z).all() and np.std(z) > 0.3:
            return f
    except Exception:
        pass
    med = np.median(x_cal)
    mad = np.median(np.abs(x_cal - med)) * 1.4826 or 1.0
    lm, ls = np.log1p(max(med, 0.0)), max(np.log1p(max(med + mad, 0.0)) - np.log1p(max(med, 0.0)), 1e-9)
    return lambda x: (np.log1p(np.maximum(np.asarray(x, np.float64), 0.0)) - lm) / ls

def _quad_expand(Z):
    Z = np.atleast_2d(Z)
    cols = [Z, Z ** 2 - 1.0]
    d = Z.shape[1]
    for i in range(d):
        for j in range(i + 1, d):
            cols.append((Z[:, i] * Z[:, j])[:, None])
    return np.hstack(cols)

def _robust_log_z(X_cal):
    L = np.log1p(np.maximum(np.atleast_2d(np.asarray(X_cal, np.float64)), 0.0))
    med = np.median(L, 0)
    mad = np.median(np.abs(L - med), 0) * 1.4826
    mad = np.where(mad < 1e-12, 1.0, mad)
    def f(X):
        return (np.log1p(np.maximum(np.atleast_2d(np.asarray(X, np.float64)), 0.0)) - med) / mad
    return f

def fit_density_scorer(X_cal, X_tune, feat_names, verbose=True):
    """
    Fit all candidate one-class density models on the genuine calibration
    split; select by mean log-likelihood on the genuine tune split.
    Returns (score_batch_fn, info). score_batch_fn maps an (n,d) array to
    anomaly scores (higher = more anomalous).
    """
    X_cal = np.atleast_2d(np.asarray(X_cal, np.float64))
    X_tune = np.atleast_2d(np.asarray(X_tune, np.float64))
    d = X_cal.shape[1]
    cands = {}

    # -- gauss: robust-z log1p + LW Gaussian (your previous system, baseline) --
    rz = _robust_log_z(X_cal)
    g0 = _GaussScorer(rz(X_cal))
    cands["gauss"] = dict(ll=float(g0.log_lik(rz(X_tune)).mean()),
                          score=lambda X, g0=g0, rz=rz: g0.maha(rz(X)))

    # -- Johnson-Gaussianized marginals --
    fs = [_johnson_gaussianize(X_cal[:, j]) for j in range(d)]
    def J(X):
        X = np.atleast_2d(np.asarray(X, np.float64))
        return np.column_stack([fs[j](X[:, j]) for j in range(d)])
    Jc, Jt = J(X_cal), J(X_tune)

    g1 = _GaussScorer(Jc)
    cands["johnson"] = dict(ll=float(g1.log_lik(Jt).mean()),
                            score=lambda X, g1=g1, J=J: g1.maha(J(X)))

    if d >= 2:
        g2 = _GaussScorer(_quad_expand(Jc))
        cands["johnson_quad"] = dict(ll=float(g2.log_lik(_quad_expand(Jt)).mean()),
                                     score=lambda X, g2=g2, J=J: g2.maha(_quad_expand(J(X))))

    for k in GMM_KS:
        if len(Jc) < 20 * k:
            continue
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             reg_covar=1e-6, random_state=SEED).fit(Jc)
        cands[f"j_gmm_{k}"] = dict(ll=float(gm.score_samples(Jt).mean()),
                                   score=lambda X, gm=gm, J=J: -gm.score_samples(J(X)))

    best = max(cands, key=lambda k: cands[k]["ll"])
    if FORCE_MODEL is not None:
        if FORCE_MODEL in cands:
            best = FORCE_MODEL
        else:
            print(f"[MODEL SELECTION] FORCE_MODEL={FORCE_MODEL} unavailable for this run "
                  f"(insufficient samples or 1-D quad); falling back to best by likelihood.")
    if verbose:
        print("[MODEL SELECTION] held-out GENUINE log-lik per candidate (label-free):")
        for name, c in sorted(cands.items(), key=lambda kv: -kv[1]["ll"]):
            print(f"    {name:<13}: {c['ll']:+.4f}" + ("  <-- selected" if name == best else ""))

    info = {"model": best,
            "heldout_ll": {k: v["ll"] for k, v in cands.items()},
            "gaussianize": J,          # for per-channel contribution printout
            "feat_names": list(feat_names)}
    return cands[best]["score"], info


# ================= MAIN ================= #
def main():
    rng = np.random.default_rng(SEED)

    T = load_module(TRAIN_SCRIPT)
    blob = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model = T.DinoRadModisAE(use_modis=True).to(device)
    model.load_state_dict(blob["state_dict"], strict=False)
    model.eval()
    size = T.ENCODER_SIZE

    # 1. Gather + deterministic splits
    all_imgs = []
    for r, _, fs in os.walk(DATA_DIR):
        for f in fs:
            if f.lower().endswith(IMG_EXTS):
                all_imgs.append(os.path.join(r, f))
    all_imgs.sort()
    rng.shuffle(all_imgs)
    selected_imgs = all_imgs[:max(1, int(len(all_imgs) * SAMPLE_FRAC))]

    all_genuine = sorted(p for p in selected_imgs if not is_ground_truth_manipulated(p))
    all_manipulated = sorted(p for p in selected_imgs if is_ground_truth_manipulated(p))
    rng.shuffle(all_genuine); rng.shuffle(all_manipulated)

    block = max(1, int(len(all_genuine) * 0.10))
    sn2_ref_imgs = set(all_genuine[:block])
    tune_imgs    = set(all_genuine[block:2 * block])
    eval_imgs    = set(all_genuine[2 * block:] + all_manipulated)

    print(f"[INFO] Data Routing Summary (seed = {SEED}):")
    print(f"       -> Calibration (sn2_ref): {len(sn2_ref_imgs)} (100% Genuine)")
    print(f"       -> Tune Set (tune_10):   {len(tune_imgs)} (100% Genuine)")
    print(f"       -> Eval Set (eval_10):   {len(eval_imgs)} "
          f"({len(all_genuine[2*block:])} Genuine / {len(all_manipulated)} Manipulated)")

    # 2. Extract raw error vectors ONCE
    rows = []
    for p in tqdm(selected_imgs, desc="Extracting error features"):
        try:
            img = load_image(p, size)
            rgb = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
            radio = torch.zeros(1, T.RADIO_SPATIAL_GRID, T.RADIO_SPATIAL_GRID, T.RADIO_DIM, device=device)
            modis = torch.tensor([0], device=device)
            errs = extract_errors(model, rgb, radio, modis)
            if errs is None:
                print("[SKIP]", p, "model output not tuple/list"); continue
            rec_err, latent_err, aux_err = errs

            if p in sn2_ref_imgs: split = "sn2_ref"
            elif p in tune_imgs:  split = "tune_10"
            elif p in eval_imgs:  split = "eval_10"
            else:                 split = "unused"

            rows.append({"image": p, "split": split, "rec_err": rec_err,
                         "latent_err": latent_err, "aux_err": aux_err,
                         "ground_truth": "manipulated" if is_ground_truth_manipulated(p) else "genuine"})
        except Exception as e:
            print("[SKIP]", p, e)

    if not rows:
        raise RuntimeError("No images were successfully processed.")
    eval_rows_all = [r for r in rows if r["split"] == "eval_10"]

    # 3-5. Ablation runs
    summary, full_contrib = [], None
    for run_name, idxs in FEATURE_SETS.items():
        feat_names = [CHANNEL_NAMES[i] for i in idxs]
        print("\n" + "#" * 96)
        print(f"# RUN: {run_name}  (features: {', '.join(feat_names)})")
        print("#" * 96)

        def fv(r): return [[r["rec_err"], r["latent_err"], r["aux_err"]][i] for i in idxs]
        def usable(r): return not any(np.isnan(x) for x in fv(r))

        X_cal = np.array([fv(r) for r in rows if r["split"] == "sn2_ref" and usable(r)])
        X_tun = np.array([fv(r) for r in rows if r["split"] == "tune_10" and usable(r)])
        if len(X_cal) < 30 or len(X_tun) < 10:
            print(f"[SKIP RUN] {run_name}: insufficient usable calibration/tune samples."); continue

        score_batch, info = fit_density_scorer(X_cal, X_tun, feat_names)

        score_key, label_key = f"score_{run_name}", f"label_{run_name}"
        use_rows = [r for r in rows if usable(r)]
        S = score_batch(np.array([fv(r) for r in use_rows]))
        for r, s in zip(use_rows, S): r[score_key] = float(s)
        for r in rows: r.setdefault(score_key, np.nan)

        # --- TUNE threshold ---
        tune_scores = [r[score_key] for r in rows if r["split"] == "tune_10" and not np.isnan(r[score_key])]
        thr = float(np.percentile(tune_scores, 100 * (1 - TARGET_FPR)))
        print(f"[TUNING] Threshold at {100*(1-TARGET_FPR):.0f}th percentile of "
              f"genuine tune scores ({info['model']}): {thr:.4f}")

        for r in rows:
            if r["split"] == "sn2_ref" or np.isnan(r[score_key]):
                r[label_key] = "genuine"
            else:
                r[label_key] = "manipulated" if r[score_key] > thr else "genuine"

        # --- EVALUATE ---
        eval_rows = [r for r in eval_rows_all if not np.isnan(r[score_key])]
        if not eval_rows: continue
        yb = np.array([1 if r["ground_truth"] == "manipulated" else 0 for r in eval_rows])
        sc = np.array([r[score_key] for r in eval_rows])
        y_true = [r["ground_truth"] for r in eval_rows]
        y_pred = [r[label_key] for r in eval_rows]

        two = len(np.unique(yb)) == 2
        auroc = roc_auc_score(yb, sc) if two else np.nan
        ap = average_precision_score(yb, sc) if two else np.nan
        bal = balanced_accuracy_score(yb, [1 if q == "manipulated" else 0 for q in y_pred])
        acc = 100.0 * np.mean([q == t for q, t in zip(y_pred, y_true)])
        gm_, mm_ = (yb == 0), (yb == 1)
        fpr = float(np.mean(sc[gm_] > thr)) if gm_.any() else np.nan
        tpr = float(np.mean(sc[mm_] > thr)) if mm_.any() else np.nan

        print(f"[EVALUATION] model={info['model']} | AUROC={auroc:.4f} | AP={ap:.4f} | "
              f"TPR@thr={tpr:.4f} | observed FPR={fpr:.4f} (target {TARGET_FPR}) | "
              f"BalancedAcc={bal:.4f} | Acc={acc:.2f}%")
        print("\n" + "=" * 25 + f" CLASSIFICATION REPORT ({run_name}, EVAL SPLIT) " + "=" * 25)
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))
        print("=" * 96)

        # --- CONTRIBUTION: mean |z| per Gaussianized channel on manipulated eval ---
        man = [r for r in eval_rows if r["ground_truth"] == "manipulated"]
        if man:
            Zm = info["gaussianize"](np.array([fv(r) for r in man]))
            mean_abs_z = np.abs(Zm).mean(0)
            contrib = {feat_names[j]: float(mean_abs_z[j]) for j in range(len(feat_names))}
            print("[CONTRIBUTION] Mean |z| (Johnson space) on manipulated eval: "
                  + ", ".join(f"{k}={v:.3f}" for k, v in contrib.items()))
            if run_name == "full": full_contrib = contrib

        summary.append({"run": run_name, "features": "+".join(feat_names),
                        "selected_model": info["model"], "threshold": thr,
                        "auroc": auroc, "ap": ap, "tpr_at_thr": tpr, "obs_fpr": fpr,
                        "balanced_acc": bal, "eval_accuracy_pct": acc})

    # --- SUMMARY + MARGINAL VALUE ---
    if summary:
        print("\n" + "=" * 40 + " ABLATION SUMMARY " + "=" * 40)
        print(f"{'Run':<12} {'Features':<24} {'Model':<12} {'AUROC':>7} {'AP':>7} "
              f"{'TPR':>6} {'FPR':>6} {'BalAcc':>7} {'Acc%':>7}")
        print("-" * 100)
        for s in summary:
            print(f"{s['run']:<12} {s['features']:<24} {s['selected_model']:<12} "
                  f"{s['auroc']:>7.4f} {s['ap']:>7.4f} {s['tpr_at_thr']:>6.3f} "
                  f"{s['obs_fpr']:>6.3f} {s['balanced_acc']:>7.4f} {s['eval_accuracy_pct']:>7.2f}")
        print("=" * 100)

        by = {s["run"]: s for s in summary}
        if "full" in by:
            print("\n[MARGINAL VALUE] AUROC gain of full over leave-one-out pairs:")
            for chan, wo in (("rec_err", "latent_aux"), ("latent_err", "rec_aux"), ("aux_err", "rec_latent")):
                if wo in by and not np.isnan(by[wo]["auroc"]):
                    d = by["full"]["auroc"] - by[wo]["auroc"]
                    verdict = "contributes" if d > 0.005 else ("neutral" if d > -0.005 else "HURTS fusion")
                    print(f"    +{chan:<12}: dAUROC = {d:+.4f}  ({verdict})")
        if full_contrib:
            weak = [k for k, v in full_contrib.items() if v < 0.5]
            if weak:
                print(f"\n[NOTE] Channels with mean |z| < 0.5 on fakes in the full model: {weak}.")

    # 6. Export
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    if summary:
        with open(CSV_OUT.replace(".csv", "_summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader(); w.writerows(summary)
    print("DONE ->", CSV_OUT)


if __name__ == "__main__":
    main()
