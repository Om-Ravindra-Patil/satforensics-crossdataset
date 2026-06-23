import os, torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DinoRadModisAE(use_modis=USE_MODIS).to(device)
train_imgs, val_imgs = genuine_split()
train_ds = TiffGenuineDataset(train_imgs)
val_ds   = TiffGenuineDataset(val_imgs)
train_loader = _make_loader(train_ds, TRAIN_WORKERS, shuffle=True)
val_loader   = _make_loader(val_ds,   VAL_WORKERS,   shuffle=False)


ae_path   = os.path.join(CHECKPOINT_DIR, "best_ae.pth")
flow_path = os.path.join(CHECKPOINT_DIR, "best_flow.pth")
model.load_state_dict(torch.load(ae_path,   map_location=device), strict=False)
model.flow.load_state_dict(torch.load(flow_path, map_location=device))
model.eval()
print(f"restored AE   <- {ae_path}")
print(f"restored flow <- {flow_path}")

print("Fitting latent Gaussian...")
gaussian = fit_latent_gaussian(model, train_loader, device)
print("Calibrating threshold on genuine val (all 4 signals)...")
calib = calibrate(model, gaussian, val_loader, device)

torch.save({'state_dict': model.state_dict(), **calib}, FINAL_CKPT_NAME)
print(f"\nSaved -> {FINAL_CKPT_NAME} | threshold={calib['threshold']:.3f}")

if os.path.exists(TEST_IMAGE):
    res = classify_image(TEST_IMAGE, model, calib, device, heatmap_path="demo_heatmap.png")
    print("\n=== TEST IMAGE ===")
    for k, v in res.items():
        print(f"  {k:12s}: {v}")
