import os
os.environ['CUDA_DEVICE_ORDER']       = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']    = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,expandable_segments:True'
os.environ['OMP_NUM_THREADS']         = '8'
os.environ['MKL_NUM_THREADS']         = '8'
os.environ['OPENBLAS_NUM_THREADS']    = '8'

import glob
import math
import hashlib
import pickle
import random
import threading
import multiprocessing as mp
from collections import OrderedDict
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tifffile
import timm
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

MODIS_PATH      = '/home/c3068579/Documents/wholeworld_epsg4326.tif'
STAGE2_DIR      = '/home/c3068579/Documents/data_dir/stage_2'
RADIO_CACHE_DIR = '/home/c3068579/Documents/radio_cache_dino'
DINO_CACHE_DIR  = '/home/c3068579/Documents/dino_token_cache'
TEST_IMAGE      = '/home/c3068579/Documents/inference_results_vc10/DiffusionSat/outputs_fmow_real/France_00000_tower_48.9836_1.8046.tif'  # optional demo

CHECKPOINT_DIR  = "checkpoints_dino_flow"
FINAL_CKPT_NAME = "satforensics_dino_film_flow.pth"

# stage toggles
DO_STAGE1_AE   = True
DO_STAGE2_FLOW = True
DO_DEMO_INFER  = True
PREVIEW_EACH_EPOCH = True
RESUME         = True
RESUME_CKPT    = None


USE_DINO_CACHE    = False
DINO_EXTRACT_BATCH = 32

# MODIS
USE_MODIS         = True
NUM_MODIS_CLASSES = 18


DINO_MODEL   = 'vit_large_patch16_dinov3.sat493m'   # satellite-pretrained ViT-L/16
ENCODER_SIZE = 256 
PATCH_PX     = 16
TOKEN_GRID   = ENCODER_SIZE // PATCH_PX
N_TOKENS     = TOKEN_GRID * TOKEN_GRID
EMBED_DIM    = 1024 
WORK_CH      = 256

DECODE_SIZES = [(ENCODER_SIZE // 8, ENCODER_SIZE // 8),
                (ENCODER_SIZE // 4, ENCODER_SIZE // 4),
                (ENCODER_SIZE // 2, ENCODER_SIZE // 2),
                (ENCODER_SIZE, ENCODER_SIZE)]

RADIO_SPATIAL_GRID = 4
RADIO_MASK_PROB    = 0.15

# flow
FLOW_BLOCKS = 8
FLOW_HIDDEN = 256
FLOW_CLAMP  = 2.0

BATCH_SIZE       = 16
GRAD_ACCUM_STEPS = 2
TRAIN_WORKERS    = 8
VAL_WORKERS      = 4
PRECOMPUTE_WORKERS = 8
PREFETCH_FACTOR  = 2
RADIO_CACHE_MAX_IMAGES = 64
PIN_MEMORY       = False

EPOCHS       = 20    # stage 1
FLOW_EPOCHS  = 15    # stage 2
BASE_LR      = 1e-4
FLOW_LR      = 2e-4
MIN_LR       = 1e-6
TRAIN_SPLIT  = 0.90

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

RADIO_KEYS = [
    "B_entropy", "B_mean", "B_sat_hi", "B_sat_lo", "B_std",
    "G_entropy", "G_mean", "G_sat_hi", "G_sat_lo", "G_std",
    "R_entropy", "R_mean", "R_sat_hi", "R_sat_lo", "R_std",
    "corr_GB", "corr_RB", "corr_RG",
    "fft_row_peak", "fft_col_peak", "mad_hp",
    "noise_slope", "noise_intercept", "noise_r2", "prnu_ratio",
]
RADIO_DIM_RAW = len(RADIO_KEYS)
RADIO_DIM     = RADIO_DIM_RAW * 2

IMAGE_TOPK_FRAC = 0.10
TARGET_FPR      = 0.05
MIN_TILE_STD    = 0.01
SCORE_COMBINE   = "sum"


assert WORK_CH % 2 == 0, "flow coupling needs even channels"

modis_src = rasterio.open(MODIS_PATH) if USE_MODIS else None


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _worker_init(worker_id):
    cv2.setNumThreads(0); torch.set_num_threads(1)
    np.random.seed(42 + worker_id)


def compute_adaptive_positions(size, win):
    if size <= win:
        return [0]
    stride = win // 2
    pos = list(range(0, size - win + 1, stride))
    if pos[-1] != size - win:
        pos.append(size - win)
    return pos


IMG_EXTS = ('.tif', '.tiff', '.jp2')


def find_all_images(root_dir):
    imgs = []
    for root, _, files in os.walk(root_dir):
        for f in sorted(files):
            low = f.lower()
            stem = os.path.splitext(low)[0]
            if low.endswith(IMG_EXTS) and not stem.endswith('_mask'):
                imgs.append(os.path.join(root, f))
    return imgs


def genuine_split():
    imgs = find_all_images(STAGE2_DIR)
    rng = random.Random(42); rng.shuffle(imgs)
    k = int(len(imgs) * TRAIN_SPLIT)
    return imgs[:k], imgs[k:]


def get_modis_majority_class(win, src):
    data = src.read(1, window=win, boundless=True, fill_value=0)
    vals, counts = np.unique(data[data > 0], return_counts=True)
    return int(vals[np.argmax(counts)]) if len(vals) > 0 else 0


def modis_class_for_patch(img_path, x, y, patch_size):
    if not USE_MODIS or modis_src is None:
        return 0
    try:
        with rasterio.open(img_path) as im:
            win = Window(x, y, patch_size, patch_size)
            bounds = rasterio.windows.bounds(win, im.transform)
            b_modis = transform_bounds(im.crs, modis_src.crs, *bounds, densify_pts=21)
        mwin = from_bounds(*b_modis, transform=modis_src.transform)
        return get_modis_majority_class(mwin, modis_src)
    except Exception:
        return 0


def compute_radiometric_features_gpu(rgb_batch: torch.Tensor) -> torch.Tensor:
    B, _, H, W = rgb_batch.shape
    device = rgb_batch.device
    flat   = rgb_batch.view(B, 3, -1)
    mean   = flat.mean(dim=-1)
    std    = flat.std(dim=-1, unbiased=False) + 1e-8
    sat_hi = (flat > 0.999).float().mean(dim=-1)
    sat_lo = (flat < 0.001).float().mean(dim=-1)

    entropy = torch.zeros(B, 3, device=device)
    for c in range(3):
        h = torch.histc(flat[:, c].clamp(0., 1.), bins=64)
        p = h / (h.sum() + 1e-8)
        entropy[:, c] = -(p * (p + 1e-12).log()).sum()

    x = (flat - mean.unsqueeze(-1)) / std.unsqueeze(-1)
    corr = torch.bmm(x, x.transpose(1, 2)) / (H * W)

    gray = rgb_batch.mean(dim=1)
    P    = torch.abs(torch.fft.fft2(gray)) ** 2
    fft_row_peak = P.mean(dim=2).max(dim=1)[0] / (P.mean(dim=2).median(dim=1)[0] + 1e-6)
    fft_col_peak = P.mean(dim=1).max(dim=1)[0] / (P.mean(dim=1).median(dim=1)[0] + 1e-6)

    down  = F.avg_pool2d(gray.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
    blur  = F.interpolate(down.unsqueeze(1), size=(H, W),
                          mode='bilinear', align_corners=False).squeeze(1)
    hp    = gray - blur
    mad_hp     = torch.median(torch.abs(hp.view(B, -1)), dim=1)[0]
    prnu_ratio = torch.var(hp, dim=(1, 2)) / (torch.var(gray, dim=(1, 2)) + 1e-8)

    block = 8
    if H >= block * 2 and W >= block * 2:
        gh, gw = H // block, W // block
        gray_c = gray[:, :gh * block, :gw * block]
        blocks = gray_c.reshape(B, gh, block, gw, block).permute(0, 1, 3, 2, 4)
        blocks = blocks.reshape(B, gh * gw, block * block)
        m = blocks.mean(-1)
        s = blocks.std(-1) + 1e-8
        m_mean = m.mean(-1, keepdim=True)
        s_mean = s.mean(-1, keepdim=True)
        cov    = ((m - m_mean) * (s - s_mean)).mean(-1)
        var_m  = ((m - m_mean) ** 2).mean(-1) + 1e-8
        slope  = cov / var_m
        inter  = s_mean.squeeze(-1) - slope * m_mean.squeeze(-1)
        pred   = slope.unsqueeze(-1) * m + inter.unsqueeze(-1)
        ss_res = ((s - pred) ** 2).sum(-1)
        ss_tot = ((s - s_mean) ** 2).sum(-1) + 1e-8
        r2     = (1 - ss_res / ss_tot).clamp(-1, 1)
    else:
        slope = torch.zeros(B, device=device)
        inter = torch.zeros(B, device=device)
        r2    = torch.zeros(B, device=device)

    feat = {
        "B_entropy": entropy[:, 2], "G_entropy": entropy[:, 1], "R_entropy": entropy[:, 0],
        "B_mean":    mean[:, 2],    "G_mean":    mean[:, 1],    "R_mean":    mean[:, 0],
        "B_sat_hi":  sat_hi[:, 2],  "G_sat_hi":  sat_hi[:, 1],  "R_sat_hi":  sat_hi[:, 0],
        "B_sat_lo":  sat_lo[:, 2],  "G_sat_lo":  sat_lo[:, 1],  "R_sat_lo":  sat_lo[:, 0],
        "B_std":     std[:, 2],     "G_std":     std[:, 1],     "R_std":     std[:, 0],
        "corr_GB":   corr[:, 1, 2], "corr_RB":   corr[:, 0, 2], "corr_RG":  corr[:, 0, 1],
        "fft_row_peak":    fft_row_peak,
        "fft_col_peak":    fft_col_peak,
        "mad_hp":          mad_hp,
        "noise_slope":     slope,
        "noise_intercept": inter,
        "noise_r2":        r2,
        "prnu_ratio":      prnu_ratio,
    }
    return torch.stack([feat[k] for k in RADIO_KEYS], dim=1)


def compute_patch_radio_grid(patch_np: np.ndarray, grid: int, device) -> np.ndarray:
    H, W = patch_np.shape[:2]
    ch, cw = max(H // grid, 1), max(W // grid, 1)
    cells = []
    for r in range(grid):
        for c in range(grid):
            y0, y1 = r * ch, (r + 1) * ch if r < grid - 1 else H
            x0, x1 = c * cw, (c + 1) * cw if c < grid - 1 else W
            cell = patch_np[y0:y1, x0:x1]
            if cell.size == 0 or cell.shape[0] < 4 or cell.shape[1] < 4:
                cell = np.zeros((ch, cw, 3), dtype=np.float32)
            cells.append(cell)
    H_max = max(c.shape[0] for c in cells)
    W_max = max(c.shape[1] for c in cells)
    padded = []
    for cell in cells:
        if cell.shape[0] != H_max or cell.shape[1] != W_max:
            pad = np.zeros((H_max, W_max, 3), dtype=np.float32)
            pad[:cell.shape[0], :cell.shape[1]] = cell
            cell = pad
        padded.append(cell)
    cells_t = torch.from_numpy(np.stack(padded)).permute(0, 3, 1, 2).float().to(device).clamp(0., 1.)
    with torch.no_grad():
        vecs = compute_radiometric_features_gpu(cells_t)
    return vecs.view(grid, grid, RADIO_DIM_RAW).cpu().numpy().astype(np.float32)


def compute_image_radio_baseline(img_np: np.ndarray, device) -> np.ndarray:
    H, W = img_np.shape[:2]
    if max(H, W) > 1024:
        s = 1024 / max(H, W)
        img_np = cv2.resize(img_np, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device).clamp(0., 1.)
    with torch.no_grad():
        return compute_radiometric_features_gpu(t).cpu().numpy().astype(np.float32)[0]



def _cache_path_for_image(img_path):
    key = hashlib.md5(img_path.encode()).hexdigest()
    name = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(RADIO_CACHE_DIR, f"{name}_{key}.pkl")


def load_radio_cache(img_path):
    p = _cache_path_for_image(img_path)
    if not os.path.exists(p):
        return None
    with open(p, 'rb') as f:
        return pickle.load(f)


def save_radio_cache(img_path, cache):
    os.makedirs(RADIO_CACHE_DIR, exist_ok=True)
    with open(_cache_path_for_image(img_path), 'wb') as f:
        pickle.dump(cache, f, pickle.HIGHEST_PROTOCOL)


def _imread_raw(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff'):
        try:
            return tifffile.imread(path)
        except Exception:
            pass
    # JP2 needs GDAL's OpenJPEG driver (rasterio). Also a fallback for odd TIFFs.
    try:
        with rasterio.open(path) as src:
            arr = src.read()                 # (bands, H, W)
        return np.transpose(arr, (1, 2, 0))  # -> (H, W, bands)
    except Exception:
        pass
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Cannot read image: {path}")
    if arr.ndim == 3:
        arr = arr[..., ::-1]                 # BGR -> RGB
    return arr


def _to_float01(img):
    """Scale to [0,1] by inferred bit-depth (handles 8/12/16-bit and float inputs).
    NOTE: Sentinel-2-style reflectance JP2s store small values in uint16; if results
    look too dark, switch to percentile scaling here."""
    img = np.asarray(img).astype(np.float32)
    mx = float(img.max()) if img.size else 1.0
    if   mx <= 1.5:     denom = 1.0
    elif mx <= 255.0:   denom = 255.0
    elif mx <= 4095.0:  denom = 4095.0      # 12-bit
    elif mx <= 65535.0: denom = 65535.0     # 16-bit
    else:               denom = mx
    return np.clip(img / denom, 0.0, 1.0)


def _load_image_for_radio(img_path):
    img = _imread_raw(img_path)
    if img is None:
        raise RuntimeError(f"Cannot read {img_path}")
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3:
        if img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[1]:
            img = np.transpose(img, (1, 2, 0))
        if img.shape[2] > 3:
            img = img[..., :3]
        if img.shape[2] == 1:
            img = np.concatenate([img] * 3, axis=-1)
    return _to_float01(img)


def _precompute_single(img_path):
    try:
        cache = load_radio_cache(img_path)
        if cache and 'baseline' in cache and any(isinstance(k, tuple) for k in cache):
            return 0
        img = _load_image_for_radio(img_path)
        h, w = img.shape[:2]

        device = torch.device('cpu')
        cache = {'baseline': compute_image_radio_baseline(img, device)}
        ys = compute_adaptive_positions(h, ENCODER_SIZE)
        xs = compute_adaptive_positions(w, ENCODER_SIZE)
        for y in ys:
            for x in xs:
                ye, xe = min(y + ENCODER_SIZE, h), min(x + ENCODER_SIZE, w)
                patch = img[y:ye, x:xe]
                if patch.shape[0] < 16 or patch.shape[1] < 16:
                    continue
                cache[(x, y, ENCODER_SIZE)] = compute_patch_radio_grid(patch, RADIO_SPATIAL_GRID, device)
        if len([k for k in cache if isinstance(k, tuple)]) == 0:
            cache[(0, 0, -1)] = compute_patch_radio_grid(img, RADIO_SPATIAL_GRID, device)
        save_radio_cache(img_path, cache)
        del img
        return 1
    except Exception as e:
        print(f"Precompute failed {img_path}: {e}")
        return -1


def precompute_radio_for_images(img_paths, desc="Precomputing"):

    from multiprocessing.pool import ThreadPool
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    with ThreadPool(PRECOMPUTE_WORKERS) as pool:
        results = list(tqdm(pool.imap_unordered(_precompute_single, img_paths, chunksize=4),
                            total=len(img_paths), desc=desc))
    print(f"Precompute: {sum(1 for r in results if r == 1)} done, "
          f"{sum(1 for r in results if r == 0)} skipped, "
          f"{sum(1 for r in results if r == -1)} failed")


@torch.no_grad()
def precompute_dino_tokens(model, img_paths, device, desc="Precompute DINOv2 tokens"):
    """Compute the FROZEN DINOv2 token grid [768,16,16] once per tile, on GPU, batched,
    and cache to disk (fp16). The backbone never changes, so all later epochs reuse this."""
    os.makedirs(DINO_CACHE_DIR, exist_ok=True)
    model.eval()
    jobs = []  # (img_path, x, y, ps, cache_path)
    for ip in img_paths:
        try:
            for (x, y, ps) in _tile_list_for_image(ip):
                cp = _dino_cache_path(ip, x, y, ps)
                if not os.path.exists(cp):
                    jobs.append((ip, x, y, ps, cp))
        except Exception as e:
            print(f"  tile-list failed {ip}: {e}")
    if not jobs:
        print(f"{desc}: all tiles already cached.")
        return
    print(f"{desc}: {len(jobs)} tiles to compute "
          f"(~{len(jobs) * model.encoder.embed_dim * TOKEN_GRID * TOKEN_GRID * 2 / 1e9:.1f} GB on disk, fp16)")

    buf_rgb, buf_cp = [], []
    def _flush():
        if not buf_rgb:
            return
        xb = torch.stack(buf_rgb).to(device)
        with autocast(device_type='cuda', dtype=torch.float32):
            grid = model.encoder._tokens_to_grid(xb)   # [B,768,16,16]
        grid = grid.half().cpu().numpy()
        for k, cp in enumerate(buf_cp):
            np.save(cp, grid[k])
        buf_rgb.clear(); buf_cp.clear()

    for (ip, x, y, ps, cp) in tqdm(jobs, desc=desc):
        try:
            patch, _, _ = _read_patch_windowed(ip, x, y, ps)
            if patch.shape[:2] != (ENCODER_SIZE, ENCODER_SIZE):
                patch = cv2.resize(patch, (ENCODER_SIZE, ENCODER_SIZE), interpolation=cv2.INTER_LINEAR)
            buf_rgb.append(_norm_rgb(patch)); buf_cp.append(cp)
            if len(buf_rgb) >= DINO_EXTRACT_BATCH:
                _flush()
        except Exception as e:
            print(f"  DINOv2 cache failed {ip} ({x},{y}): {e}")
    _flush()


@torch.no_grad()
def benchmark_dino_split(model, loader, device, iters=5):
    """Print the measured DINOv2-forward vs rest-of-step split for THIS hardware."""
    import time
    try:
        batch = next(iter(loader))
    except StopIteration:
        return
    rgb   = batch[0].to(device)
    modis = batch[1].to(device)
    radio = batch[6].to(device)
    dino  = batch[7].to(device) if batch[7] is not None else None
    model.eval()
    sync = (lambda: torch.cuda.synchronize()) if device.type == 'cuda' else (lambda: None)

    sync(); t0 = time.time()
    for _ in range(iters):
        _ = model.encoder._tokens_to_grid(rgb)
    sync(); t_dino = (time.time() - t0) / iters

    sync(); t0 = time.time()
    for _ in range(iters):
        s = model.encode(rgb, radio, modis, dino_grid=dino)
        _ = model.decoder(s)
    sync(); t_rest = (time.time() - t0) / iters

    print(f"[bench] DINOv2 forward/batch: {t_dino*1000:.1f} ms | "
          f"encode+decode/batch (cache={'ON' if dino is not None else 'OFF'}): {t_rest*1000:.1f} ms")
    if t_dino + t_rest > 0:
        share = t_dino / (t_dino + t_rest) * 100
        print(f"[bench] DINOv2 is ~{share:.0f}% of the uncached forward → caching skips that each epoch")


class RadioNormalizer(nn.Module):
    def __init__(self, dim=RADIO_DIM, momentum=0.01):
        super().__init__()
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var',  torch.ones(dim))
        self.register_buffer('n_updates',    torch.zeros(1))
        self.momentum = momentum

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                flat = x.reshape(-1, x.shape[-1])
                mean = flat.mean(0)
                var  = flat.var(0, unbiased=False) + 1e-8
                if self.n_updates.item() == 0:
                    self.running_mean.copy_(mean); self.running_var.copy_(var)
                else:
                    self.running_mean.mul_(1 - self.momentum).add_(mean * self.momentum)
                    self.running_var.mul_(1 - self.momentum).add_(var * self.momentum)
                self.n_updates += 1
        out = (x - self.running_mean) / torch.sqrt(self.running_var + 1e-6)
        return torch.clamp(out, -5, 5)


class LRURadioCache:
    def __init__(self, max_images: int = RADIO_CACHE_MAX_IMAGES):
        self._max = max_images
        self._store: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, img_path: str):
        with self._lock:
            if img_path in self._store:
                self._store.move_to_end(img_path)
                return self._store[img_path]
        cache = load_radio_cache(img_path)
        if cache is None:
            raise RuntimeError(f"No radio cache on disk for {img_path}")
        with self._lock:
            self._store[img_path] = cache
            self._store.move_to_end(img_path)
            while len(self._store) > self._max:
                self._store.popitem(last=False)
        return cache


def _read_patch_windowed(img_path, x, y, patch_size):
    try:
        with rasterio.open(img_path) as src:
            h, w = src.height, src.width
            xs = min(x, w - patch_size) if patch_size != -1 else 0
            ys = min(y, h - patch_size) if patch_size != -1 else 0
            ps = patch_size if patch_size != -1 else max(h, w)
            win = Window(xs, ys, min(ps, w - xs), min(ps, h - ys))
            data = src.read(window=win)
            if data.shape[0] == 1:
                data = np.concatenate([data] * 3, axis=0)
            elif data.shape[0] > 3:
                data = data[:3]
            patch = _to_float01(data.transpose(1, 2, 0))
            return patch, xs, ys
    except Exception:
        img = _imread_raw(img_path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3:
            if img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[1]:
                img = img.transpose(1, 2, 0)
            if img.shape[2] > 3:
                img = img[..., :3]
            if img.shape[2] == 1:
                img = np.concatenate([img] * 3, axis=-1)
        img = _to_float01(img)
        h, w = img.shape[:2]
        if patch_size == -1:
            return img, 0, 0
        xs = min(x, w - patch_size); ys = min(y, h - patch_size)
        return img[ys:ys+patch_size, xs:xs+patch_size], xs, ys


def _radio_from_cache(cache, x, y, patch_size):
    key = (x, y, patch_size)
    if key not in cache:
        key = (0, 0, -1) if (0, 0, -1) in cache else next(k for k in cache if isinstance(k, tuple))
    abs_grid = cache[key]
    residual = abs_grid - cache['baseline'][None, None, :]
    return torch.from_numpy(np.concatenate([abs_grid, residual], axis=-1)).float()


def _norm_rgb(patch):
    rgb_t = torch.from_numpy(patch).permute(2, 0, 1).float()
    return (rgb_t - IMAGENET_MEAN) / IMAGENET_STD


def _dino_cache_path(img_path, x, y, ps):
    key = hashlib.md5(f"{img_path}|{x}|{y}|{ps}".encode()).hexdigest()
    return os.path.join(DINO_CACHE_DIR, f"{key}.npy")


def _image_hw(img_path):
    try:
        with rasterio.open(img_path) as src:
            return src.height, src.width
    except Exception:
        im = _imread_raw(img_path)
        if im.ndim == 3 and im.shape[0] in (1, 3, 4) and im.shape[0] < im.shape[1]:
            im = im.transpose(1, 2, 0)
        return im.shape[0], im.shape[1]


def _tile_list_for_image(img_path):
    """Tile positions for an image — MUST match TiffGenuineDataset exactly."""
    h, w = _image_hw(img_path)
    ys = compute_adaptive_positions(h, ENCODER_SIZE)
    xs = compute_adaptive_positions(w, ENCODER_SIZE)
    tiles = [(x, y, ENCODER_SIZE) for y in ys for x in xs]
    if not tiles:
        tiles = [(0, 0, -1)]
    return tiles


class TiffGenuineDataset(Dataset):
    def __init__(self, img_paths):
        self.samples = []
        self._cache = LRURadioCache()
        for img_path in tqdm(img_paths, desc="Building dataset"):
            try:
                try:
                    with rasterio.open(img_path) as src:
                        h, w = src.height, src.width
                except Exception:
                    im = _imread_raw(img_path)
                    if im.ndim == 3 and im.shape[0] in (1, 3, 4) and im.shape[0] < im.shape[1]:
                        im = im.transpose(1, 2, 0)
                    h, w = im.shape[:2]; del im
                ys = compute_adaptive_positions(h, ENCODER_SIZE)
                xs = compute_adaptive_positions(w, ENCODER_SIZE)
                fn = os.path.splitext(os.path.basename(img_path))[0]
                for y in ys:
                    for x in xs:
                        self.samples.append((fn, img_path, x, y, ENCODER_SIZE))
                if not ys or not xs:
                    self.samples.append((fn, img_path, 0, 0, -1))
            except Exception:
                continue
        random.shuffle(self.samples)
        print(f"Dataset tiles: {len(self.samples)}")

    def __getitem__(self, idx):
        for _ in range(5):
            try:
                fn, img_path, x, y, ps = self.samples[idx]
                patch, xs, ys = _read_patch_windowed(img_path, x, y, ps)
                if patch.shape[:2] != (ENCODER_SIZE, ENCODER_SIZE):
                    patch = cv2.resize(patch, (ENCODER_SIZE, ENCODER_SIZE), interpolation=cv2.INTER_LINEAR)
                rgb_t = _norm_rgb(patch)
                mc = modis_class_for_patch(img_path, x, y, ENCODER_SIZE)
                modis_t = torch.tensor(max(0, min(mc, NUM_MODIS_CLASSES - 1)), dtype=torch.long)
                radio = _radio_from_cache(self._cache.get(img_path), x, y, ps)
                if USE_DINO_CACHE:
                    dino = torch.from_numpy(np.load(_dino_cache_path(img_path, x, y, ps)))  # fp16 [768,16,16]
                else:
                    dino = torch.empty(0)
                return rgb_t, modis_t, fn, x, y, ps, radio, dino
            except Exception:
                idx = (idx + 1) % len(self.samples)
        return (torch.zeros(3, ENCODER_SIZE, ENCODER_SIZE), torch.tensor(0),
                "__pad__", 0, 0, ENCODER_SIZE,
                torch.zeros(RADIO_SPATIAL_GRID, RADIO_SPATIAL_GRID, RADIO_DIM),
                torch.empty(0))

    def __len__(self):
        return len(self.samples)


def genuine_collate(batch):
    rgb   = torch.stack([b[0] for b in batch])
    modis = torch.stack([b[1] for b in batch])
    fn    = [b[2] for b in batch]
    xs    = torch.tensor([b[3] for b in batch], dtype=torch.long)
    ys    = torch.tensor([b[4] for b in batch], dtype=torch.long)
    psz   = torch.tensor([b[5] for b in batch], dtype=torch.long)
    radio = torch.stack([b[6] for b in batch])
    dino  = torch.stack([b[7] for b in batch]) if batch[0][7].numel() > 0 else None
    return rgb, modis, fn, xs, ys, psz, radio, dino


def _make_loader(ds, num_workers, shuffle):
    kw = dict(batch_size=BATCH_SIZE, shuffle=shuffle, pin_memory=PIN_MEMORY,
              num_workers=num_workers, collate_fn=genuine_collate)
    if num_workers and num_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=PREFETCH_FACTOR,
                  worker_init_fn=_worker_init)
    return DataLoader(ds, **kw)



class SpatialRadioFiLM(nn.Module):
    def __init__(self, feature_channels, radio_dim=RADIO_DIM, hidden_dim=128):
        super().__init__()
        self.radio_net = nn.Sequential(
            nn.Conv2d(radio_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True))
        self.gamma_head = nn.Conv2d(hidden_dim, feature_channels, 1, bias=True)
        self.beta_head  = nn.Conv2d(hidden_dim, feature_channels, 1, bias=True)
        nn.init.zeros_(self.gamma_head.weight); nn.init.ones_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight);  nn.init.zeros_(self.beta_head.bias)

    def forward(self, feat, radio_grid):
        radio_up = F.interpolate(radio_grid.float(), size=feat.shape[-2:],
                                 mode='bilinear', align_corners=False)
        h = self.radio_net(radio_up)
        gamma = torch.clamp(self.gamma_head(h), 0.5, 2.0)
        return gamma * feat + self.beta_head(h)


class DinoV2FiLMEncoder(nn.Module):
    def __init__(self, use_modis=True, radio_dim=RADIO_DIM, work_ch=WORK_CH, modis_dim=256):
        super().__init__()
        self.use_modis = use_modis
        self.backbone = timm.create_model(
            DINO_MODEL, pretrained=True, num_classes=0,
            img_size=ENCODER_SIZE, dynamic_img_size=True)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.num_prefix = getattr(self.backbone, 'num_prefix_tokens', 1)
        self.embed_dim  = self.backbone.num_features
        self.proj = nn.Sequential(
            nn.Conv2d(self.embed_dim, work_ch, 1, bias=False),
            nn.BatchNorm2d(work_ch), nn.ReLU(inplace=True))
        self.radio_film  = SpatialRadioFiLM(work_ch, radio_dim)
        self.modis_gamma = nn.Linear(modis_dim, work_ch)
        self.modis_beta  = nn.Linear(modis_dim, work_ch)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _tokens_to_grid(self, x):
        tok = self.backbone.forward_features(x)
        tok = tok[:, self.num_prefix:, :]
        B, N, D = tok.shape
        g = int(round(math.sqrt(N)))
        return tok.transpose(1, 2).reshape(B, D, g, g)

    def forward(self, rgb=None, radio_grid=None, modis_token=None, dino_grid=None):
        if dino_grid is not None:
            grid = dino_grid.to(self.proj[0].weight.device, non_blocking=True).float()
        else:
            with torch.no_grad():
                grid = self._tokens_to_grid(rgb)
        f = self.proj(grid)
        if radio_grid is not None:
            f = self.radio_film(f, radio_grid)
        if modis_token is not None:
            gamma = self.modis_gamma(modis_token).unsqueeze(-1).unsqueeze(-1)
            beta  = self.modis_beta(modis_token).unsqueeze(-1).unsqueeze(-1)
            f = f * gamma + beta
        return f


class GridDecoder(nn.Module):
    def __init__(self, in_ch=WORK_CH, sizes=DECODE_SIZES):
        super().__init__()
        self.sizes = sizes
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.b1 = block(in_ch, 256); self.b2 = block(256, 128)
        self.b3 = block(128, 64);    self.b4 = block(64, 64)
        self.head = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        x = F.interpolate(self.b1(x), size=self.sizes[0], mode='bilinear', align_corners=False)
        x = F.interpolate(self.b2(x), size=self.sizes[1], mode='bilinear', align_corners=False)
        x = F.interpolate(self.b3(x), size=self.sizes[2], mode='bilinear', align_corners=False)
        x = F.interpolate(self.b4(x), size=self.sizes[3], mode='bilinear', align_corners=False)
        return self.head(x)


class ActNorm(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(1, ch, 1, 1))
        self.bias      = nn.Parameter(torch.zeros(1, ch, 1, 1))
        self.register_buffer('inited', torch.zeros(1))

    def forward(self, x):
        if self.inited.item() == 0 and self.training:
            with torch.no_grad():
                mean = x.mean(dim=[0, 2, 3], keepdim=True)
                std  = x.std(dim=[0, 2, 3], keepdim=True) + 1e-6
                self.log_scale.data.copy_(torch.log(1.0 / std))
                self.bias.data.copy_(-mean / std)
                self.inited.fill_(1.0)
        y = x * torch.exp(self.log_scale) + self.bias
        H, W = x.shape[-2:]
        ld = self.log_scale.sum().expand(x.shape[0], H, W)  # per-pixel logdet
        return y, ld


class AffineCoupling(nn.Module):
    def __init__(self, ch, hidden=FLOW_HIDDEN, clamp=FLOW_CLAMP):
        super().__init__()
        self.clamp = clamp
        half = ch // 2
        self.net = nn.Sequential(
            nn.Conv2d(half, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, (ch - half) * 2, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        x_a, x_b = x.chunk(2, dim=1)
        log_s, t = self.net(x_a).chunk(2, dim=1)
        log_s = self.clamp * torch.tanh(log_s)
        y_b = x_b * torch.exp(log_s) + t
        y = torch.cat([x_a, y_b], dim=1)
        ld = log_s.sum(dim=1)  # [B,H,W]
        return y, ld


class FastFlow2D(nn.Module):
    def __init__(self, ch=WORK_CH, n_blocks=FLOW_BLOCKS, hidden=FLOW_HIDDEN):
        super().__init__()
        self.acts  = nn.ModuleList([ActNorm(ch) for _ in range(n_blocks)])
        self.coups = nn.ModuleList([AffineCoupling(ch, hidden) for _ in range(n_blocks)])

    def forward(self, x):
        ld_total = x.new_zeros(x.shape[0], x.shape[-2], x.shape[-1])
        for act, coup in zip(self.acts, self.coups):
            x, ld1 = act(x);  ld_total = ld_total + ld1
            x, ld2 = coup(x); ld_total = ld_total + ld2
            x = x.flip(1)
        return x, ld_total

    def nll_map(self, x):
        z, ld = self.forward(x)
        return 0.5 * (z ** 2).sum(dim=1) - ld


class DinoRadModisAE(nn.Module):
    def __init__(self, use_modis=True):
        super().__init__()
        self.use_modis      = use_modis
        self.radio_norm     = RadioNormalizer(RADIO_DIM)
        self.modis_embed    = nn.Embedding(NUM_MODIS_CLASSES, 256)
        self.encoder        = DinoV2FiLMEncoder(use_modis, RADIO_DIM, WORK_CH, modis_dim=256)
        self.decoder        = GridDecoder(WORK_CH)
        self.aux_radio_head = nn.Sequential(
            nn.Conv2d(WORK_CH, 128, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, RADIO_DIM_RAW, 1))
        self.flow           = FastFlow2D(WORK_CH)

    def _norm_radio(self, radio_grid):
        x = self.radio_norm(radio_grid)
        return x.permute(0, 3, 1, 2).contiguous()

    def encode(self, rgb, radio_grid, modis_ids, drop_radio=False, dino_grid=None):
        modis_token = self.modis_embed(modis_ids) if self.use_modis else None
        rg = None if drop_radio else self._norm_radio(radio_grid)
        return self.encoder(rgb, rg, modis_token, dino_grid=dino_grid)

    def forward(self, rgb, radio_grid, modis_ids, drop_radio=False, dino_grid=None):
        s = self.encode(rgb, radio_grid, modis_ids, drop_radio, dino_grid=dino_grid)
        recon  = self.decoder(s)
        latent = F.adaptive_avg_pool2d(s, 1).flatten(1)
        aux    = self.aux_radio_head(s).mean(dim=(2, 3))
        return recon, latent, aux

class GaussianLatent:
    def __init__(self, shrink=0.1):
        self.shrink, self.mu, self.prec = shrink, None, None
    def fit(self, Z):
        self.mu = Z.mean(0)
        S = np.cov(Z, rowvar=False); d = S.shape[0]
        S = (1 - self.shrink) * S + self.shrink * (np.trace(S) / d) * np.eye(d)
        self.prec = np.linalg.inv(S + 1e-6 * np.eye(d)); return self
    def distance(self, Z):
        diff = Z - self.mu
        return np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', diff, self.prec, diff), 0.0))


class Standardizer:
    def __init__(self): self.med, self.scale = 0.0, 1.0
    def fit(self, x):
        self.med = float(np.median(x))
        self.scale = 1.4826 * float(np.median(np.abs(x - self.med))) + 1e-8; return self
    def transform(self, x): return (np.asarray(x) - self.med) / self.scale


def combine_tile_scores(rec_z, maha_z, radio_z, flow_z, mode):
    if mode == "max":
        return np.maximum.reduce([rec_z, maha_z, radio_z, flow_z])
    return rec_z + maha_z + radio_z + flow_z


def aggregate_image(tile_scores, topk_frac):
    ts = np.asarray(tile_scores)
    if ts.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(topk_frac * ts.size)))
    return float(np.sort(ts)[-k:].mean())


@torch.no_grad()
def raw_tile_signals(model, rgb, radio_grid, modis_ids, gaussian, device, dino_grid=None):
    """Returns rec, maha, radio, flow (per-tile), tile_std, and flow NLL token-maps."""
    model.eval()
    with autocast(device_type='cuda', dtype=torch.float32):
        s = model.encode(rgb, radio_grid, modis_ids, drop_radio=False, dino_grid=dino_grid)
        recon  = model.decoder(s)
        latent = F.adaptive_avg_pool2d(s, 1).flatten(1)
        aux    = model.aux_radio_head(s).mean(dim=(2, 3))
        nll    = model.flow.nll_map(s.float())               # [B,16,16]
    rec_err = (recon - rgb).abs().mean(dim=(1, 2, 3)).cpu().numpy()
    rg_norm = model.radio_norm(radio_grid)[..., :RADIO_DIM_RAW].mean(dim=(1, 2))
    radio_err = (aux.float() - rg_norm.float()).abs().mean(dim=1).cpu().numpy()
    z = latent.cpu().numpy()
    maha = gaussian.distance(z) if gaussian is not None else np.zeros(len(z))
    nll_np = nll.cpu().numpy()
    flow_tile = nll_np.reshape(nll_np.shape[0], -1).mean(axis=1)
    tile_std = rgb.flatten(1).std(dim=1).cpu().numpy()
    return rec_err, maha, radio_err, flow_tile, tile_std, nll_np


def reconstruction_loss(recon, rgb):
    return F.l1_loss(recon, rgb)


@torch.no_grad()
def evaluate_val_recon(model, val_loader, device):
    model.eval(); tot = n = 0.0
    for rgb, modis_ids, _, _, _, _, radio_grids, dino in tqdm(val_loader, desc="Validating"):
        rgb = rgb.to(device, non_blocking=True)
        modis_ids = modis_ids.to(device, non_blocking=True)
        radio_grid = radio_grids.to(device, non_blocking=True)
        dino = dino.to(device, non_blocking=True) if dino is not None else None
        with autocast(device_type='cuda', dtype=torch.float32):
            recon, _, _ = model(rgb, radio_grid, modis_ids, dino_grid=dino)
        tot += (recon - rgb).abs().mean().item() * rgb.size(0); n += rgb.size(0)
    vr = tot / max(n, 1); print(f"Val genuine recon-L1: {vr:.5f}"); return vr


@torch.no_grad()
def fit_latent_gaussian(model, loader, device, max_tiles=20000):
    model.eval(); Z = []
    for rgb, modis_ids, _, _, _, _, radio_grids, dino in tqdm(loader, desc="Latents"):
        rgb = rgb.to(device, non_blocking=True)
        modis_ids = modis_ids.to(device, non_blocking=True)
        radio_grid = radio_grids.to(device, non_blocking=True)
        dino = dino.to(device, non_blocking=True) if dino is not None else None
        with autocast(device_type='cuda', dtype=torch.float32):
            _, latent, _ = model(rgb, radio_grid, modis_ids, dino_grid=dino)
        Z.append(latent.cpu().numpy())
        if sum(len(a) for a in Z) >= max_tiles:
            break
    return GaussianLatent().fit(np.concatenate(Z, 0))


@torch.no_grad()
def calibrate(model, gaussian, val_loader, device):
    per_image: Dict[str, List[Tuple]] = {}
    rec_all, maha_all, radio_all, flow_all = [], [], [], []
    for rgb, modis_ids, fnames, _, _, _, radio_grids, dino in tqdm(val_loader, desc="Calibrating"):
        rgb = rgb.to(device, non_blocking=True)
        modis_ids = modis_ids.to(device, non_blocking=True)
        radio_grid = radio_grids.to(device, non_blocking=True)
        dino = dino.to(device, non_blocking=True) if dino is not None else None
        rec, maha, radio_err, flow, tstd, _ = raw_tile_signals(
            model, rgb, radio_grid, modis_ids, gaussian, device, dino_grid=dino)
        rec_all.append(rec); maha_all.append(maha); radio_all.append(radio_err); flow_all.append(flow)
        for j, fn in enumerate(fnames):
            per_image.setdefault(fn, []).append((rec[j], maha[j], radio_err[j], flow[j], tstd[j]))
    std_rec   = Standardizer().fit(np.concatenate(rec_all))
    std_maha  = Standardizer().fit(np.concatenate(maha_all))
    std_radio = Standardizer().fit(np.concatenate(radio_all))
    std_flow  = Standardizer().fit(np.concatenate(flow_all))
    img_scores = []
    for fn, sig in per_image.items():
        sig = np.array(sig)
        valid = sig[:, 4] >= MIN_TILE_STD
        use = sig[valid] if valid.any() else sig
        ts = combine_tile_scores(std_rec.transform(use[:, 0]), std_maha.transform(use[:, 1]),
                                 std_radio.transform(use[:, 2]), std_flow.transform(use[:, 3]),
                                 SCORE_COMBINE)
        img_scores.append(aggregate_image(ts, IMAGE_TOPK_FRAC))
    img_scores = np.array(img_scores)
    threshold = float(np.quantile(img_scores, 1.0 - TARGET_FPR))
    print(f"Genuine image score: median={np.median(img_scores):.3f} thr@FPR{TARGET_FPR}={threshold:.3f}")
    return {
        "gaussian_mu": gaussian.mu, "gaussian_prec": gaussian.prec,
        "std_rec": (std_rec.med, std_rec.scale), "std_maha": (std_maha.med, std_maha.scale),
        "std_radio": (std_radio.med, std_radio.scale), "std_flow": (std_flow.med, std_flow.scale),
        "combine": SCORE_COMBINE, "topk_frac": IMAGE_TOPK_FRAC,
        "min_tile_std": MIN_TILE_STD, "threshold": threshold,
        "use_modis": USE_MODIS, "num_modis_classes": NUM_MODIS_CLASSES,
    }


def get_lr(step, total_steps, base_lr, min_lr):
    progress = step / max(1, total_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def load_resume_state(model, optimizer, train_loader_len, device):
    if not RESUME:
        return 0, 0, float('inf'), 0
    ckpts = [RESUME_CKPT] if RESUME_CKPT else sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "epoch_*.pth")))
    if not ckpts:
        print(f"RESUME on, no checkpoints in '{CHECKPOINT_DIR}' — starting fresh.")
        return 0, 0, float('inf'), 0
    history = []
    for c in ckpts:
        try:
            d = torch.load(c, map_location='cpu', weights_only=False)
            history.append((int(d['epoch']), float(d['val_recon']), c))
        except Exception as e:
            print(f"  skip unreadable checkpoint {c}: {e}")
    if not history:
        return 0, 0, float('inf'), 0
    history.sort(key=lambda t: t[0])
    best, no_improve = float('inf'), 0
    for ep, vr, _ in history:
        if vr < best: best, no_improve = vr, 0
        else: no_improve += 1
    last_ep, _, last_path = history[-1]
    ck = torch.load(last_path, map_location=device, weights_only=False)
    model.load_state_dict(ck['state_dict'], strict=False)
    try:
        optimizer.load_state_dict(ck['optimizer'])
    except Exception:
        print("  optimizer mismatch — fresh optimizer.")
    start_epoch = last_ep; global_step = start_epoch * train_loader_len
    print(f"\n── RESUMED {last_path} | epochs={start_epoch} best={best:.5f}\n")
    return start_epoch, global_step, best, no_improve

def train_unsup_ae(model, train_loader, val_loader, device):
    print("\n=== STAGE 1: unsupervised AE (genuine only) ===")
    ae_params = [p for n, p in model.named_parameters()
                 if p.requires_grad and not n.startswith("flow.")]
    optimizer = optim.AdamW(ae_params, lr=BASE_LR, weight_decay=1e-4)
    scaler = GradScaler()
    total_steps = EPOCHS * len(train_loader); patience = 4
    start_epoch, global_step, best_recon, no_improve = load_resume_state(
        model, optimizer, len(train_loader), device)

    for epoch in range(start_epoch, EPOCHS):
        model.train(); optimizer.zero_grad(set_to_none=True)
        for step, (rgb, modis_ids, _, _, _, _, radio_grids, dino) in enumerate(
                tqdm(train_loader, desc=f"[S1] Epoch {epoch+1}/{EPOCHS}")):
            rgb = rgb.to(device, non_blocking=True)
            modis_ids = modis_ids.to(device, non_blocking=True)
            radio_grid = radio_grids.to(device, non_blocking=True)
            dino = dino.to(device, non_blocking=True) if dino is not None else None
            lr = get_lr(global_step, total_steps, BASE_LR, MIN_LR)
            for pg in optimizer.param_groups: pg['lr'] = lr
            global_step += 1
            drop_radio = random.random() < RADIO_MASK_PROB
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                recon, latent, aux = model(rgb, radio_grid, modis_ids, drop_radio=drop_radio, dino_grid=dino)
                loss = reconstruction_loss(recon, rgb)
                if not drop_radio:
                    with torch.no_grad():
                        target = model.radio_norm(radio_grid)[..., :RADIO_DIM_RAW].mean(dim=(1, 2))
                    loss = loss + 0.1 * F.smooth_l1_loss(aux, target.float())
                loss = loss / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ae_params, 1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)

        val_recon = evaluate_val_recon(model, val_loader, device)
        preview_test_image(model, device, tag=f"s1_ep{epoch+1:02d}", use_flow=False)
        ckpt = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1:02d}_vr{val_recon:.5f}.pth")
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'val_recon': float(val_recon)}, ckpt)
        if val_recon < best_recon:
            best_recon = val_recon; no_improve = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_ae.pth"))
            print(f"New best recon: {best_recon:.5f}")
        else:
            no_improve += 1; print(f"No improvement ({no_improve}/{patience}) best={best_recon:.5f}")
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}"); break

    bp = os.path.join(CHECKPOINT_DIR, "best_ae.pth")
    if os.path.exists(bp):
        model.load_state_dict(torch.load(bp, map_location=device), strict=False)


@torch.no_grad()
def evaluate_val_flow(model, val_loader, device):
    model.eval(); tot = n = 0.0
    for rgb, modis_ids, _, _, _, _, radio_grids, dino in tqdm(val_loader, desc="Val flow NLL"):
        rgb = rgb.to(device, non_blocking=True)
        modis_ids = modis_ids.to(device, non_blocking=True)
        radio_grid = radio_grids.to(device, non_blocking=True)
        dino = dino.to(device, non_blocking=True) if dino is not None else None
        s = model.encode(rgb, radio_grid, modis_ids, dino_grid=dino)
        nll = model.flow.nll_map(s.float()).mean()
        tot += nll.item() * rgb.size(0); n += rgb.size(0)
    v = tot / max(n, 1); print(f"Val flow NLL: {v:.4f}"); return v


def train_flow(model, train_loader, val_loader, device):
    print("\n=== STAGE 2: unsupervised normalizing flow (genuine, frozen features) ===")
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.flow.parameters():
        p.requires_grad_(True)
    optimizer = optim.AdamW(model.flow.parameters(), lr=FLOW_LR, weight_decay=1e-5)
    total_steps = FLOW_EPOCHS * len(train_loader)
    best_nll, gstep = float('inf'), 0

    for epoch in range(FLOW_EPOCHS):
        model.eval(); model.flow.train()          # frozen encoder, trainable flow
        optimizer.zero_grad(set_to_none=True)
        for rgb, modis_ids, _, _, _, _, radio_grids, dino in tqdm(
                train_loader, desc=f"[S2] Epoch {epoch+1}/{FLOW_EPOCHS}"):
            rgb = rgb.to(device, non_blocking=True)
            modis_ids = modis_ids.to(device, non_blocking=True)
            radio_grid = radio_grids.to(device, non_blocking=True)
            dino = dino.to(device, non_blocking=True) if dino is not None else None
            lr = get_lr(gstep, total_steps, FLOW_LR, MIN_LR)
            for pg in optimizer.param_groups: pg['lr'] = lr
            gstep += 1
            with torch.no_grad():
                s = model.encode(rgb, radio_grid, modis_ids, dino_grid=dino).float()
            # flows are sensitive — train in fp32, no autocast
            nll = model.flow.nll_map(s).mean()
            nll.backward()
            torch.nn.utils.clip_grad_norm_(model.flow.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)

        v = evaluate_val_flow(model, val_loader, device)
        preview_test_image(model, device, tag=f"s2_ep{epoch+1:02d}", use_flow=True)
        if v < best_nll:
            best_nll = v
            torch.save(model.flow.state_dict(), os.path.join(CHECKPOINT_DIR, "best_flow.pth"))
            print(f"  new best flow NLL: {best_nll:.4f}")

    bp = os.path.join(CHECKPOINT_DIR, "best_flow.pth")
    if os.path.exists(bp):
        model.flow.load_state_dict(torch.load(bp, map_location=device))
    print(f"Stage 2 done. Best val flow NLL: {best_nll:.4f}")



@torch.no_grad()
def preview_test_image(model, device, tag, use_flow):

    if not (PREVIEW_EACH_EPOCH and os.path.exists(TEST_IMAGE)):
        return
    model.eval()
    try:
        img = _load_image_for_radio(TEST_IMAGE)
    except Exception as e:
        print(f"  [preview] could not read TEST_IMAGE: {e}")
        return
    H, W = img.shape[:2]
    baseline = compute_image_radio_baseline(img, device)
    ys = compute_adaptive_positions(H, ENCODER_SIZE)
    xs = compute_adaptive_positions(W, ENCODER_SIZE)

    heat = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    rgb_b, modis_b, radio_b, coords = [], [], [], []
    for y in ys:
        for x in xs:
            ye, xe = min(y + ENCODER_SIZE, H), min(x + ENCODER_SIZE, W)
            patch = img[y:ye, x:xe]
            if patch.shape[0] < 16 or patch.shape[1] < 16:
                continue
            ph, pw = patch.shape[:2]
            p256 = patch if patch.shape[:2] == (ENCODER_SIZE, ENCODER_SIZE) else \
                cv2.resize(patch, (ENCODER_SIZE, ENCODER_SIZE), interpolation=cv2.INTER_LINEAR)
            rt = _norm_rgb(p256)
            abs_grid = compute_patch_radio_grid(p256, RADIO_SPATIAL_GRID, device)
            full = np.concatenate([abs_grid, abs_grid - baseline[None, None, :]], -1)
            mc = modis_class_for_patch(TEST_IMAGE, x, y, ENCODER_SIZE)
            rgb_b.append(rt); modis_b.append(max(0, min(mc, NUM_MODIS_CLASSES - 1)))
            radio_b.append(torch.from_numpy(full).float()); coords.append((x, y, ye, xe, ph, pw))

    vals = []
    for i in range(0, len(rgb_b), BATCH_SIZE):
        rgb = torch.stack(rgb_b[i:i+BATCH_SIZE]).to(device)
        modis = torch.tensor(modis_b[i:i+BATCH_SIZE], dtype=torch.long, device=device)
        radio = torch.stack(radio_b[i:i+BATCH_SIZE]).to(device)
        with autocast(device_type='cuda', dtype=torch.float32):
            s = model.encode(rgb, radio, modis)
            if use_flow:
                m = model.flow.nll_map(s.float()).cpu().numpy() 
            else:
                recon = model.decoder(s)
                m = (recon - rgb).abs().mean(dim=1).cpu().numpy()
        for j in range(m.shape[0]):
            x, y, ye, xe, ph, pw = coords[i + j]
            mp_ = cv2.resize(m[j], (pw, ph), interpolation=cv2.INTER_LINEAR)
            heat[y:ye, x:xe] += mp_; cnt[y:ye, x:xe] += 1.0
            vals.append(float(mp_.mean()))

    if not vals:
        return
    heat = np.divide(heat, np.maximum(cnt, 1e-6))
    hn = heat - heat.min(); hn = hn / (hn.max() + 1e-8)
    hm = cv2.applyColorMap((hn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    rgb_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)[..., ::-1]
    out = os.path.join(CHECKPOINT_DIR, f"preview_{tag}.png")
    cv2.imwrite(out, cv2.addWeighted(rgb_u8, 0.55, hm, 0.45, 0))
    sig = "flow-NLL" if use_flow else "recon-err"
    v = np.array(vals)
    print(f"  [preview {tag}] {sig}: mean={v.mean():.4f} top10%={np.sort(v)[-max(1,len(v)//10):].mean():.4f} -> {out}")


@torch.no_grad()
def classify_image(img_path, model, calib, device, heatmap_path=None):
    g = GaussianLatent(); g.mu = calib["gaussian_mu"]; g.prec = calib["gaussian_prec"]
    sr  = Standardizer(); sr.med, sr.scale = calib["std_rec"]
    sm  = Standardizer(); sm.med, sm.scale = calib["std_maha"]
    sra = Standardizer(); sra.med, sra.scale = calib["std_radio"]
    sf  = Standardizer(); sf.med, sf.scale = calib["std_flow"]

    img = _load_image_for_radio(img_path)
    H, W = img.shape[:2]
    baseline = compute_image_radio_baseline(img, device)
    ys = compute_adaptive_positions(H, ENCODER_SIZE)
    xs = compute_adaptive_positions(W, ENCODER_SIZE)

    heat = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    rgb_b, modis_b, radio_b, coords = [], [], [], []
    for y in ys:
        for x in xs:
            ye, xe = min(y + ENCODER_SIZE, H), min(x + ENCODER_SIZE, W)
            patch = img[y:ye, x:xe]
            if patch.shape[0] < 16 or patch.shape[1] < 16:
                continue
            ph, pw = patch.shape[:2]
            p224 = patch if patch.shape[:2] == (ENCODER_SIZE, ENCODER_SIZE) else \
                cv2.resize(patch, (ENCODER_SIZE, ENCODER_SIZE), interpolation=cv2.INTER_LINEAR)
            rt = _norm_rgb(p224)
            abs_grid = compute_patch_radio_grid(p224, RADIO_SPATIAL_GRID, device)
            full = np.concatenate([abs_grid, abs_grid - baseline[None, None, :]], -1)
            mc = modis_class_for_patch(img_path, x, y, ENCODER_SIZE)
            rgb_b.append(rt); modis_b.append(max(0, min(mc, NUM_MODIS_CLASSES - 1)))
            radio_b.append(torch.from_numpy(full).float())
            coords.append((x, y, ye, xe, ph, pw))

    scores = []
    for i in range(0, len(rgb_b), BATCH_SIZE):
        rgb = torch.stack(rgb_b[i:i+BATCH_SIZE]).to(device)
        modis = torch.tensor(modis_b[i:i+BATCH_SIZE], dtype=torch.long, device=device)
        radio = torch.stack(radio_b[i:i+BATCH_SIZE]).to(device)
        rec, maha, radio_err, flow, tstd, nll_map = raw_tile_signals(model, rgb, radio, modis, g, device)
        ts = combine_tile_scores(sr.transform(rec), sm.transform(maha), sra.transform(radio_err),
                                 sf.transform(flow), calib["combine"])
        for j in range(len(ts)):
            x, y, ye, xe, ph, pw = coords[i + j]
            pmap = cv2.resize(nll_map[j], (pw, ph), interpolation=cv2.INTER_LINEAR)
            heat[y:ye, x:xe] += pmap; cnt[y:ye, x:xe] += 1.0
        valid = tstd >= calib["min_tile_std"]
        scores.append(ts[valid] if valid.any() else ts)

    scores = np.concatenate(scores) if scores else np.array([])
    image_score = aggregate_image(scores, calib["topk_frac"])
    label = "manipulated" if image_score > calib["threshold"] else "genuine"

    if heatmap_path:
        heat = np.divide(heat, np.maximum(cnt, 1e-6))
        hn = heat - heat.min(); hn = hn / (hn.max() + 1e-8)
        hm = cv2.applyColorMap((hn * 255).astype(np.uint8), cv2.COLORMAP_JET)
        rgb_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)[..., ::-1]
        cv2.imwrite(heatmap_path, cv2.addWeighted(rgb_u8, 0.55, hm, 0.45, 0))

    return {"image": img_path, "label": label, "image_score": image_score,
            "threshold": calib["threshold"], "margin": image_score - calib["threshold"]}



def run():
    print("\n#### SatForensics: DINOv3 + Radio/MODIS FiLM + Normalizing Flow (unsupervised) ####")
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RADIO_CACHE_DIR, exist_ok=True)

    train_imgs, val_imgs = genuine_split()
    print(f"Genuine train: {len(train_imgs)} | val: {len(val_imgs)}")
    if not train_imgs:
        raise SystemExit(f"No .tif/.tiff under {STAGE2_DIR}")
    precompute_radio_for_images(train_imgs, "Precompute train radio")
    precompute_radio_for_images(val_imgs,   "Precompute val radio")

    model = DinoRadModisAE(use_modis=USE_MODIS).to(device)
    n_tr = sum(p.numel() for n, p in model.named_parameters()
               if p.requires_grad and not n.startswith('encoder.backbone'))
    print(f"Trainable (non-backbone) params: {n_tr/1e6:.2f}M | DINOv3 frozen")

    if USE_DINO_CACHE:
        precompute_dino_tokens(model, train_imgs, device, "Precompute DINOv2 (train)")
        precompute_dino_tokens(model, val_imgs,   device, "Precompute DINOv2 (val)")

    train_ds = TiffGenuineDataset(train_imgs)
    val_ds   = TiffGenuineDataset(val_imgs)
    train_loader = _make_loader(train_ds, TRAIN_WORKERS, shuffle=True)
    val_loader   = _make_loader(val_ds,   VAL_WORKERS,   shuffle=False)

    benchmark_dino_split(model, train_loader, device)

    if DO_STAGE1_AE:
        train_unsup_ae(model, train_loader, val_loader, device)
    else:
        bp = os.path.join(CHECKPOINT_DIR, "best_ae.pth")
        if os.path.exists(bp):
            model.load_state_dict(torch.load(bp, map_location=device), strict=False)
            print(f"Loaded AE weights from {bp}")

    if DO_STAGE2_FLOW:
        train_flow(model, train_loader, val_loader, device)
    else:
        bp = os.path.join(CHECKPOINT_DIR, "best_flow.pth")
        if os.path.exists(bp):
            model.flow.load_state_dict(torch.load(bp, map_location=device))
            print(f"Loaded flow weights from {bp}")

    print("\nFitting latent Gaussian + calibrating (all 4 signals)...")
    gaussian = fit_latent_gaussian(model, train_loader, device)
    calib = calibrate(model, gaussian, val_loader, device)
    torch.save({'state_dict': model.state_dict(), **calib}, FINAL_CKPT_NAME)
    print(f"Saved deployable detector → {FINAL_CKPT_NAME} | threshold={calib['threshold']:.3f}")

    if DO_DEMO_INFER and os.path.exists(TEST_IMAGE):
        res = classify_image(TEST_IMAGE, model, calib, device, heatmap_path="demo_heatmap.png")
        print("\n=== DEMO INFERENCE ===")
        for k, v in res.items():
            print(f"  {k:12s}: {v}")
        print("  heatmap     : demo_heatmap.png")


if __name__ == "__main__":
    run()
