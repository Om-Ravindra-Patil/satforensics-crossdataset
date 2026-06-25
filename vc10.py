#vc10
import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pywt
import tifffile
import timm
import rasterio
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm
import math
import random
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def get_majority_class(geo_tiff_path, jp2_folder):

    with rasterio.open(geo_tiff_path) as src_tiff:
        tiff_crs = src_tiff.crs
        nodata = src_tiff.nodata
        results = {}

        for filename in os.listdir(jp2_folder):
            if not filename.endswith('.jp2'):
                continue
            jp2_path = os.path.join(jp2_folder, filename)
            key = os.path.splitext(filename)[0]

            with rasterio.open(jp2_path) as src_jp2:
                jp2_bounds = src_jp2.bounds
                jp2_crs = src_jp2.crs

                if jp2_crs is None:
                    print(f"Warning: {filename} has no CRS; assigning class -1.")
                    results[key] = -1
                    continue

                if jp2_crs != tiff_crs:
                    try:
                        jp2_bounds = rasterio.warp.transform_bounds(
                            jp2_crs, tiff_crs, *jp2_bounds
                        )
                    except Exception as e:
                        print(f"Warning: failed to reproject {filename}: {e}. Assigning class -1.")
                        results[key] = -1
                        continue

                cx = (jp2_bounds[0] + jp2_bounds[2]) / 2.0
                cy = (jp2_bounds[1] + jp2_bounds[3]) / 2.0
                try:
                    val = next(src_tiff.sample([(cx, cy)]))[0]
                except Exception as e:
                    print(f"Warning: could not sample {filename}: {e}. Assigning class -1.")
                    results[key] = -1
                    continue

                if nodata is not None and val == nodata:
                    print(f"Warning: {filename} centre is nodata in class raster; assigning class -1.")
                    results[key] = -1
                    continue

                results[key] = int(val)

    return results


class SafeDiceLoss(nn.Module):
    def __init__(self, smooth=1e-7):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (inputs * targets).sum(dim=1)
        union = inputs.sum(dim=1) + targets.sum(dim=1)

        dice = torch.where(
            union > 0,
            (2. * intersection + self.smooth) / (union + self.smooth),
            torch.ones_like(union)
        )
        return 1 - dice.mean()


class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, pos_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.dice = SafeDiceLoss()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, inputs, targets):
        self.bce.pos_weight = self.bce.pos_weight.to(inputs.device)
        dice_loss = self.dice(inputs, targets)
        bce_loss = self.bce(inputs, targets)
        return self.alpha * dice_loss + (1 - self.alpha) * bce_loss


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        TP = (inputs * targets).sum(dim=1)
        FP = ((1 - targets) * inputs).sum(dim=1)
        FN = (targets * (1 - inputs)).sum(dim=1)

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        loss = torch.pow((1 - Tversky), self.gamma)
        return loss.mean()


def extract_haar(image):
    """SWT2 'haar' level-1 detail coeffs per RGB channel -> H x W x 9."""
    coeffs = []
    for i in range(3):
        cA, (cH, cV, cD) = pywt.swt2(image[..., i], 'haar', level=1)[0]
        coeffs.append(np.stack([cH, cV, cD], axis=-1))
    haar = np.concatenate(coeffs, axis=-1)
    return haar


def extract_laplacian_pyramid(image, levels=1):
    """Laplacian residual per RGB channel -> H x W x 3."""
    pyramid = []
    for c in range(3):  # rgb
        current = image[..., c].copy()
        for _ in range(levels):
            down = cv2.pyrDown(current)
            up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
            lap = cv2.subtract(current, up)
            pyramid.append(lap)
            current = down
    while len(pyramid) < 3:
        pyramid.append(pyramid[-1])
    laplacian = np.stack(pyramid[:3], axis=-1)
    return laplacian


def extract_srm(image):
    """SRM residual filters on the first channel -> H x W x 3."""
    filters = np.array([
        [[0, 0, 0, 0, 0],
         [0, -1, 2, -1, 0],
         [0, 2, -4, 2, 0],
         [0, -1, 2, -1, 0],
         [0, 0, 0, 0, 0]],
        [[-1, 2, -2, 2, -1],
         [2, -6, 8, -6, 2],
         [-2, 8, -12, 8, -2],
         [2, -6, 8, -6, 2],
         [-1, 2, -2, 2, -1]],
        [[0, 0, 0, 0, 0],
         [0, 1, -2, 1, 0],
         [0, -2, 4, -2, 0],
         [0, 1, -2, 1, 0],
         [0, 0, 0, 0, 0]]
    ], dtype=np.float32)
    srm_maps = []
    for f in filters:
        filtered = cv2.filter2D(image[..., 0], -1, f)
        srm_maps.append(filtered)
    srm = np.stack(srm_maps, axis=-1)
    return srm


def compute_pos_weight(mask_paths):
    total_pos = 0
    total_neg = 0
    for path in mask_paths:
        mask = tifffile.imread(path)
        if mask.ndim > 2:
            mask = mask[..., 0]
        mask = (mask > 127).astype(np.uint8)
        total_pos += mask.sum()
        total_neg += (mask == 0).sum()
    return total_neg / (total_pos + 1e-7)


# Masks now live in the SAME folder as the training images.
mask_paths = glob.glob("/home/c3068579/Documents/vcip_dup/img2/split/train/images/*_mask.tif")
pos_weight = compute_pos_weight(mask_paths)
print(f"Suggested pos_weight: {pos_weight:.2f}")


def reconstruct_and_evaluate(model, val_loader, device, patch_size=128, stride=64):
    sample_index = 0
    model.eval()
    enable_mc_dropout(model)

    filenames_all = sorted(set(f for f, _, _, _ in val_loader.dataset.samples))
    # Masks sit alongside the images in the dataset directory.
    mask_dir = val_loader.dataset.mask_dir

    image_predictions = {}
    image_targets = {}
    weight_masks = {}

    for filename in filenames_all:
        mask_path = os.path.join(mask_dir, f"{filename}_mask.tif")
        mask = tifffile.imread(mask_path)
        if mask.ndim > 2:
            mask = mask[..., 0]
        h, w = mask.shape
        image_predictions[filename] = np.zeros((h, w), dtype=np.float32)
        image_targets[filename] = np.zeros((h, w), dtype=np.float32)
        weight_masks[filename] = np.zeros((h, w), dtype=np.float32)

    gaussian_kernel = cv2.getGaussianKernel(patch_size, patch_size / 6)
    gaussian_window = gaussian_kernel @ gaussian_kernel.T
    gaussian_window = gaussian_window / gaussian_window.max()

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)

        for rgb, haar, srm, lap, mask_patch, class_id, x, y in progress_bar:
            batch_size = rgb.size(0)

            for i in range(batch_size):
                if sample_index >= len(val_loader.dataset.samples):
                    continue

                filename, x_val, y_val, _ = val_loader.dataset.samples[sample_index]
                sample_index += 1
                x_val = int(x_val)
                y_val = int(y_val)

                input_rgb = rgb[i].unsqueeze(0).to(device)
                input_haar = haar[i].unsqueeze(0).to(device)
                input_srm = srm[i].unsqueeze(0).to(device)
                input_lap = lap[i].unsqueeze(0).to(device)
                input_class_id = class_id[i].unsqueeze(0).to(device)
                input_mask = mask_patch[i].squeeze().cpu().numpy()

                mc_preds = []
                for _ in range(5):
                    with autocast(device_type='cuda', dtype=torch.float16):
                        output = model(input_rgb, input_haar, input_srm, input_lap, input_class_id)
                        pred = torch.sigmoid(output).squeeze().cpu().numpy()
                        mc_preds.append(pred)
                pred = np.mean(mc_preds, axis=0)
                weighted_pred = pred * gaussian_window

                image_predictions[filename][y_val:y_val+patch_size, x_val:x_val+patch_size] += weighted_pred
                weight_masks[filename][y_val:y_val+patch_size, x_val:x_val+patch_size] += gaussian_window
                image_targets[filename][y_val:y_val+patch_size, x_val:x_val+patch_size] = input_mask

    total_pred = []
    total_gt = []
    for filename in filenames_all:
        normalized_pred = image_predictions[filename] / (weight_masks[filename] + 1e-8)
        pred_binary = (normalized_pred > 0.5).astype(np.uint8)
        gt = (image_targets[filename] > 0.5).astype(np.uint8)
        total_pred.append(pred_binary.flatten())
        total_gt.append(gt.flatten())

    total_pred = np.concatenate(total_pred)
    total_gt = np.concatenate(total_gt)

    fg_intersection = np.sum((total_pred == 1) & (total_gt == 1))
    fg_union = np.sum(total_pred == 1) + np.sum(total_gt == 1)
    fg_dice = (2. * fg_intersection + 1e-7) / (fg_union + 1e-7)

    bg_intersection = np.sum((total_pred == 0) & (total_gt == 0))
    bg_union = np.sum(total_pred == 0) + np.sum(total_gt == 0)
    bg_dice = (2. * bg_intersection + 1e-7) / (bg_union + 1e-7)

    print("\n=== Final Overall Dice Scores ===")
    print(f"Foreground (manipulated) Dice:   {fg_dice:.4f}")
    print(f"Background (unmanipulated) Dice: {bg_dice:.4f}")

    return fg_dice, bg_dice


class JP2MaskDataset(Dataset):
    """
    Reads RGB patches from .jp2 images and computes haar / laplacian / srm
    features on the fly (no HDF5). Masks are expected in the SAME folder as
    the images, named '{filename}_mask.tif'.
    """
    def __init__(self, image_paths, mask_dir, patch_size=128, class_info=None,
                 manipulated_only_ratio=0.5, missing_class_id=0):
        self.patch_size = patch_size
        self.samples = []
        self.class_info = class_info if class_info is not None else {}
        self.stride = patch_size // 2
        self.mask_dir = mask_dir
        self.manipulated_only_ratio = manipulated_only_ratio
        self.missing_class_id = missing_class_id  # fallback for unknown filenames
        self.manipulated_samples = []
        self.normal_samples = []
        self.image_lookup = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}

        for img_path in image_paths:
            filename = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(mask_dir, f"{filename}_mask.tif")
            if not os.path.exists(mask_path):
                print(f"Warning: mask for {filename} not found in {mask_dir}; skipping.")
                continue

            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            h, w = image.shape[:2]

            mask = tifffile.imread(mask_path)
            if mask.ndim > 2:
                mask = mask[..., 0]

            for y in range(0, h - patch_size + 1, self.stride):
                for x in range(0, w - patch_size + 1, self.stride):
                    patch = mask[y:y+patch_size, x:x+patch_size]
                    sample = (filename, x, y, mask_path)
                    if (patch > 127).sum() > 0:
                        self.manipulated_samples.append(sample)
                    else:
                        self.normal_samples.append(sample)

        self.samples = self.manipulated_samples + self.normal_samples
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, x, y, mask_path = self.samples[idx]

        rgb_full = cv2.imread(self.image_lookup[filename])
        if rgb_full is None:
            new_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(new_idx)
        rgb_full = cv2.cvtColor(rgb_full, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb_patch = np.ascontiguousarray(rgb_full[y:y+self.patch_size, x:x+self.patch_size])

        mask = tifffile.imread(mask_path)
        if mask.ndim > 2:
            mask = mask[..., 0]
        mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]

        haar_np = extract_haar(rgb_patch)              # 128 x 128 x 9
        lap_np = extract_laplacian_pyramid(rgb_patch)  # 128 x 128 x 3
        srm_np = extract_srm(rgb_patch)                # 128 x 128 x 3

        rgb_tensor = torch.from_numpy(rgb_patch).permute(2, 0, 1).float()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        rgb_tensor = (rgb_tensor - mean) / std

        haar = torch.from_numpy(np.ascontiguousarray(haar_np)).permute(2, 0, 1).float()
        lap = torch.from_numpy(np.ascontiguousarray(lap_np)).permute(2, 0, 1).float()
        srm = torch.from_numpy(np.ascontiguousarray(srm_np)).permute(2, 0, 1).float()


        mask_tensor = torch.from_numpy((mask_patch > 127).astype(np.float32)).unsqueeze(0)
        class_id = self.class_info.get(filename, self.missing_class_id)
        class_id = torch.tensor(class_id, dtype=torch.long)

        return rgb_tensor, haar, srm, lap, mask_tensor, class_id, x, y


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x)))
        x3 = F.relu(self.bn3(self.conv3(x)))
        x4 = F.relu(self.bn4(self.conv4(x)))
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        return self.project(x_cat)


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        resnet = timm.create_model('resnet34d', pretrained=True)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.act1,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )

    def forward(self, x):
        return self.encoder(x)


class GatedSWPACrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        self.class_proj = nn.Linear(dim, dim)

        self.output_conv_spatial = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.output_conv_channel = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        self.gate_sa = nn.Parameter(torch.tensor(0.5))
        self.gate_ca = nn.Parameter(torch.tensor(0.5))

    def forward(self, q_input, kv_input, class_token=None):
        B, C, H, W = q_input.shape

        if class_token is not None:
            class_map = self.class_proj(class_token).unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)
            q_input = q_input + class_map
            kv_input = kv_input + class_map

        Q = self.q_proj(q_input)
        K = self.k_proj(kv_input)
        V = self.v_proj(kv_input)

        q = Q.view(B, self.heads, C // self.heads, H * W).permute(0, 1, 3, 2)
        k = K.view(B, self.heads, C // self.heads, H * W)
        attn = torch.softmax(torch.matmul(q, k) * self.scale, dim=-1)

        v = V.view(B, self.heads, C // self.heads, H * W).permute(0, 1, 3, 2)
        out_s = torch.matmul(attn, v).permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        out_s = self.output_conv_spatial(out_s)

        q_c = Q.view(B, C, -1)
        k_c = K.view(B, C, -1).permute(0, 2, 1)
        attn_c = torch.softmax(torch.bmm(q_c, k_c), dim=-1)
        v_c = V.view(B, C, -1)
        out_c = torch.bmm(attn_c, v_c).view(B, C, H, W)
        out_c = self.output_conv_channel(out_c)

        return q_input + self.gate_sa * out_s + self.gate_ca * out_c


class FullModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.rgb_encoder = ResNetEncoder(3)
        self.haar_encoder = ResNetEncoder(9)
        self.srm_encoder = ResNetEncoder(3)
        self.lap_encoder = ResNetEncoder(3)
        self.class_embed = nn.Embedding(num_classes, 256)
        self.class_proj = nn.Conv2d(256, 1024, 1)
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)

        self.cross = nn.ModuleDict({
            'rgb_from_others': nn.ModuleList([
                GatedSWPACrossAttentionBlock(256, 256),
                GatedSWPACrossAttentionBlock(256, 256),
                GatedSWPACrossAttentionBlock(256, 256)
            ]),
            'haar_from_others': nn.ModuleList([
                GatedSWPACrossAttentionBlock(256, 256),
                GatedSWPACrossAttentionBlock(256, 256),
                GatedSWPACrossAttentionBlock(256, 256)
            ]),
            'srm_from_others': nn.ModuleList([
                GatedSWPACrossAttentionBlock(256, 256),
                GatedSWPACrossAttentionBlock(256, 256),
                GatedSWPACrossAttentionBlock(256, 256)
            ]),
            'lap_from_others': nn.ModuleList([
                GatedSWPACrossAttentionBlock(256, 256),
                GatedSWPACrossAttentionBlock(256, 256),
                GatedSWPACrossAttentionBlock(256, 256)
            ]),
            'class_from_all': GatedSWPACrossAttentionBlock(256, 256)
        })

        self.decoder = DeepLabHead(in_channels=256 * 4, num_classes=1)

    def forward(self, rgb, haar, srm, lap, class_id):
        rgb_feat = self.rgb_encoder(rgb)
        haar_feat = self.haar_encoder(haar)
        srm_feat = self.srm_encoder(srm)
        lap_feat = self.lap_encoder(lap)
        class_token = self.class_embed(class_id)

        rgb_feat = sum([
            self.cross['rgb_from_others'][0](rgb_feat, haar_feat, class_token),
            self.cross['rgb_from_others'][1](rgb_feat, srm_feat, class_token),
            self.cross['rgb_from_others'][2](rgb_feat, lap_feat, class_token)
        ]) / 3

        haar_feat = sum([
            self.cross['haar_from_others'][0](haar_feat, rgb_feat, class_token),
            self.cross['haar_from_others'][1](haar_feat, srm_feat, class_token),
            self.cross['haar_from_others'][2](haar_feat, lap_feat, class_token)
        ]) / 3

        srm_feat = sum([
            self.cross['srm_from_others'][0](srm_feat, rgb_feat, class_token),
            self.cross['srm_from_others'][1](srm_feat, haar_feat, class_token),
            self.cross['srm_from_others'][2](srm_feat, lap_feat, class_token)
        ]) / 3

        lap_feat = sum([
            self.cross['lap_from_others'][0](lap_feat, rgb_feat, class_token),
            self.cross['lap_from_others'][1](lap_feat, haar_feat, class_token),
            self.cross['lap_from_others'][2](lap_feat, srm_feat, class_token)
        ]) / 3

        combined_token = class_token  # (B, 256)
        combined_token = combined_token.unsqueeze(2).unsqueeze(3)
        combined_token = combined_token.expand(-1, -1, rgb_feat.shape[2], rgb_feat.shape[3])
        class_feat = self.cross['class_from_all'](combined_token, rgb_feat)

        fused = torch.cat([rgb_feat, haar_feat, srm_feat, lap_feat], dim=1)
        fused = fused + self.class_proj(class_feat)
        out = self.decoder(fused)
        out = self.upsample(out)
        return out


def dice_score(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    targets = (targets > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2. * intersection + 1e-7) / (union + 1e-7)
    return dice.mean().item()


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dirs
    train_image_dir = '/home/c3068579/Documents/vcip_dup/img2/split/train/images'
    val_image_dir = '/home/c3068579/Documents/vcip_dup/img2/split/val/images'
    geo_tiff_path = '/home/c3068579/Documents/wholeworld_epsg4326.tif'

    class_info = get_majority_class(geo_tiff_path, train_image_dir)
    class_info.update(get_majority_class(geo_tiff_path, val_image_dir))

    unique_classes = sorted(set(class_info.values()))
    class_remap = {c: i for i, c in enumerate(unique_classes)}
    class_info = {fname: class_remap[c] for fname, c in class_info.items()}
    num_classes = len(class_remap)
    missing_class_id = class_remap.get(-1, 0)  # bucket for filenames not in the dict
    print(f"Class remap: {class_remap}  ->  num_classes={num_classes}")

    train_paths = glob.glob(os.path.join(train_image_dir, "*.jp2"))
    val_paths = glob.glob(os.path.join(val_image_dir, "*.jp2"))

    train_dataset = JP2MaskDataset(train_paths, train_image_dir, patch_size=128,
                                   class_info=class_info, missing_class_id=missing_class_id)
    val_dataset = JP2MaskDataset(val_paths, val_image_dir, patch_size=128,
                                 class_info=class_info, missing_class_id=missing_class_id)

    train_loader = DataLoader(train_dataset, batch_size=110, shuffle=True, num_workers=24,
                              pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=110, shuffle=False, num_workers=12,
                            pin_memory=True, persistent_workers=True)

    model = FullModel(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    criterion = ComboLoss(alpha=0.5, pos_weight=pos_weight)

    scaler = GradScaler()
    best_dice = 0.0
    num_epochs = 30
    epochs_no_improve = 0
    patience = 5

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=True)

        for rgb, haar, srm, lap, masks, class_ids, _, _ in progress_bar:
            rgb = rgb.to(device)
            haar = haar.to(device)
            srm = srm.to(device)
            lap = lap.to(device)
            masks = masks.to(device)
            class_ids = class_ids.to(device)

            optimizer.zero_grad()
            outputs = model(rgb, haar, srm, lap, class_ids)
            loss = criterion(outputs, masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}] Learning Rate: {current_lr:.6f}")
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {epoch_loss:.4f}")

        fg_dice, bg_dice = reconstruct_and_evaluate(model, val_loader, device, patch_size=128)
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation FG Dice: {fg_dice:.4f} | BG Dice: {bg_dice:.4f}")

        if fg_dice > best_dice:
            best_dice = fg_dice
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model with Foreground Dice: {best_dice:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best Dice: {best_dice:.4f}")
            break


if __name__ == "__main__":
    train()
