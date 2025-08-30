# lab4b_fno_tiny_airfrans.py
# Tiny FNO-2D on AirfRANS to predict Cp from [x,y,cos(aoa),sin(aoa),Uinf,p_inf].
# - Uses TRAIN normalization stats for both train/val
# - Adds p_inf input channel (constant field per case)
# - Robust loss: SmoothL1 + small MSE term
# - Reports val MSE (z-scored) and RMSE in Cp units
# - CPU-friendly; ~1–2M params by default

import os, math, argparse
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ---------------- AirfRANS loading & rasterization ----------------
def ensure_airfrans(root):
    import airfrans as af
    dataset_dir = os.path.join(root, "Dataset")
    if not os.path.exists(dataset_dir):
        print("[setup] Downloading AirfRANS (preprocessed, no OpenFOAM files)…")
        af.dataset.download(root=root, file_name="Dataset", unzip=True, OpenFOAM=False)
    return dataset_dir

def load_airfrans_as_rasters(root, H=128, W=128, M=64):
    """Return dict with keys: p,u,v,Xg,Yg; each rasterized to [M,H,W]."""
    import airfrans as af
    from scipy.interpolate import griddata

    dataset_dir = ensure_airfrans(root)
    data_list, _ = af.dataset.load(root=dataset_dir, task="full", train=True)
    M = min(M, len(data_list))
    print(f"[data] Using {M} snapshots from AirfRANS train split")

    xg = np.linspace(-2.0, 4.0, W, dtype="float32")
    yg = np.linspace(-1.5, 1.5, H, dtype="float32")
    Xg, Yg = np.meshgrid(xg, yg)

    fields = {"p": 9, "u": 7, "v": 8}
    out = {k: [] for k in fields}
    for i in range(M):
        sim = data_list[i]
        x, y = sim[:, 0].astype("float32"), sim[:, 1].astype("float32")
        for k, idx in fields.items():
            vals = sim[:, idx].astype("float32")
            Zi = griddata(np.c_[x, y], vals, (Xg, Yg), method="linear")
            mask = np.isnan(Zi)
            if mask.any():
                Zi[mask] = griddata(np.c_[x, y], vals, (Xg[mask], Yg[mask]), method="nearest")
            out[k].append(Zi.astype("float32"))
    for k in out:
        out[k] = np.stack(out[k], axis=0)  # [M,H,W]
    out["Xg"], out["Yg"] = Xg, Yg
    return out

# ---------------- Compute Cp and case-wise AoA/Uinf/p_inf ----------------
def compute_case_params(p, u, v, border=5):
    """
    p,u,v: [H,W] arrays (p is p/rho). Estimate far-field from outer ring.
    Returns Cp [H,W], aoa(rad), Uinf, pinf (floats).
    """
    H, W = p.shape
    mask = np.zeros_like(p, dtype=bool)
    mask[:border, :] = True; mask[-border:, :] = True
    mask[:, :border] = True; mask[:, -border:] = True

    u_inf = float(np.mean(u[mask])); v_inf = float(np.mean(v[mask]))
    p_inf = float(np.mean(p[mask]))
    U_inf = float(np.hypot(u_inf, v_inf))
    aoa = float(np.arctan2(v_inf, u_inf + 1e-12))
    denom = max(U_inf**2, 1e-6)
    Cp = 2.0 * (p - p_inf) / denom
    return Cp.astype("float32"), aoa, U_inf, p_inf

# ---------------- Dataset ----------------
class AirfRANSFnoDataset(Dataset):
    """
    Builds inputs X: [N, C_in, H, W] and targets Y: [N, 1, H, W].
    Inputs channels: [x_norm, y_norm, cos(aoa), sin(aoa), Uinf, p_inf]  -> C_in=6
    Targets: Cp.
    If stats is None: computes mean/std on this split.
    If stats provided: (x_mean,x_std,y_mean,y_std) and uses those.
    """
    def __init__(self, bundle, idxs, stats=None):
        self.p = bundle["p"][idxs]  # [N,H,W]
        self.u = bundle["u"][idxs]
        self.v = bundle["v"][idxs]
        self.Xg = bundle["Xg"]; self.Yg = bundle["Yg"]

        N, H, W = self.p.shape
        xnorm = (self.Xg - self.Xg.min()) / (self.Xg.max() - self.Xg.min()) * 2 - 1
        ynorm = (self.Yg - self.Yg.min()) / (self.Yg.max() - self.Yg.min()) * 2 - 1

        Xin, Yout = [], []
        for i in range(N):
            Cp, aoa, Uinf, p_inf = compute_case_params(self.p[i], self.u[i], self.v[i])
            aoa_cos = np.full_like(xnorm, np.cos(aoa), dtype="float32")
            aoa_sin = np.full_like(xnorm, np.sin(aoa), dtype="float32")
            Uchan   = np.full_like(xnorm, Uinf, dtype="float32")
            Pinf    = np.full_like(xnorm, p_inf, dtype="float32")
            x = np.stack([xnorm, ynorm, aoa_cos, aoa_sin, Uchan, Pinf], axis=0)  # [6,H,W]
            Xin.append(x.astype("float32"))
            Yout.append(Cp[None, ...])  # [1,H,W]
        self.X = np.stack(Xin, axis=0)
        self.Y = np.stack(Yout, axis=0)

        if stats is None:
            self.x_mean = self.X.mean(axis=(0,2,3), keepdims=True)
            self.x_std  = self.X.std(axis=(0,2,3), keepdims=True) + 1e-6
            self.y_mean = self.Y.mean(axis=(0,2,3), keepdims=True)
            self.y_std  = self.Y.std(axis=(0,2,3), keepdims=True) + 1e-6
        else:
            self.x_mean, self.x_std, self.y_mean, self.y_std = stats

        self.X = (self.X - self.x_mean) / self.x_std
        self.Y = (self.Y - self.y_mean) / self.y_std

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])

# ---------------- Tiny FNO-2D ----------------
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.weight_real = nn.Parameter(torch.randn(in_channels, out_channels, modes1, modes2) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(in_channels, out_channels, modes1, modes2) * 0.02)

    def compl_mul2d(self, a, br, bi):
        # a: [B,Cin,m1,m2], returns [B,Cout,m1,m2]
        ar, ai = a.real, a.imag
        real = torch.einsum("bixy,ioxy->boxy", ar, br) - torch.einsum("bixy,ioxy->boxy", ai, bi)
        imag = torch.einsum("bixy,ioxy->boxy", ar, bi) + torch.einsum("bixy,ioxy->boxy", ai, br)
        return torch.complex(real, imag)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")  # [B,C,H,W//2+1]
        out_ft = torch.zeros(B, self.out_channels, H, W//2 + 1, dtype=torch.complex64, device=x.device)
        m1 = min(self.modes1, H); m2 = min(self.modes2, W//2 + 1)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2],
            self.weight_real[:, :, :m1, :m2],
            self.weight_imag[:, :, :m1, :m2]
        )
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")

class FNO2dTiny(nn.Module):
    def __init__(self, in_channels=6, out_channels=1, width=48, modes1=16, modes2=16, layers=4):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.spec = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(layers)])
        self.lin  = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.head = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_channels, 1))

    def forward(self, x):
        x = self.lift(x)
        for sc, ln in zip(self.spec, self.lin):
            x = F.gelu(sc(x) + ln(x))
        return self.head(x)

# ---------------- Train / Eval ----------------
def criterion(pred, y):
    # Hybrid robust loss: Smooth L1 (Huber) + small MSE
    return F.smooth_l1_loss(pred, y, beta=0.02) + 0.1 * F.mse_loss(pred, y)

def train_one_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, device, y_std_np, y_mean_np):
    model.eval()
    mse_sum, phys_se_sum, n_pix = 0.0, 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        mse_sum += F.mse_loss(pred, yb, reduction="sum").item()
        # De-normalize to Cp units
        pred_np = pred.cpu().numpy() * y_std_np + y_mean_np
        y_np    = yb.cpu().numpy()   * y_std_np + y_mean_np
        phys_se_sum += np.sum((pred_np - y_np) ** 2)
        n_pix += yb.numel()
    mse = mse_sum / len(loader.dataset)
    rmse_phys = math.sqrt(phys_se_sum / n_pix)
    return mse, rmse_phys

@torch.no_grad()
def save_preview(model, ds, device, path, idx=0, y_std=None, y_mean=None):
    model.eval()
    xb, yb = ds[idx]
    xb = xb[None].to(device)                 # [1,C,H,W]
    pred = model(xb)[0,0].cpu().numpy()
    # back to Cp units for plotting
    y_true = (yb[0].numpy() * y_std + y_mean)
    y_pred = (pred * y_std + y_mean)
    both = np.concatenate([y_true.ravel(), y_pred.ravel()])
    vmin, vmax = np.percentile(both, 1), np.percentile(both, 99)
    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1); plt.imshow(y_true, origin="lower", vmin=vmin, vmax=vmax); plt.title("True Cp"); plt.colorbar()
    plt.subplot(1,2,2); plt.imshow(y_pred, origin="lower", vmin=vmin, vmax=vmax); plt.title("Pred Cp"); plt.colorbar()
    plt.tight_layout(); plt.savefig(path); plt.close()

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="AirfRANS")
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--W", type=int, default=128)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="lab4b_outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cpu")

    bundle = load_airfrans_as_rasters(args.root, H=args.H, W=args.W, M=args.M)
    N = bundle["p"].shape[0]

    # Shuffle indices with fixed seed to avoid ordering bias
    perm = np.random.default_rng(args.seed).permutation(N)
    Ttr = max(8, int(np.floor(args.train_frac * N)))
    train_idx, val_idx = perm[:Ttr], perm[Ttr:]

    # Build datasets; use TRAIN stats for both splits
    ds_train = AirfRANSFnoDataset(bundle, train_idx)
    train_stats = (ds_train.x_mean, ds_train.x_std, ds_train.y_mean, ds_train.y_std)
    ds_val   = AirfRANSFnoDataset(bundle, val_idx, stats=train_stats)

    tr_loader = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=0)
    va_loader = DataLoader(ds_val,   batch_size=args.batch, shuffle=False, num_workers=0)

    model = FNO2dTiny(
        in_channels=6, out_channels=1,
        width=args.width, modes1=args.modes, modes2=args.modes, layers=args.layers
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params ~ {n_params/1e6:.2f}M on {device}")

    # numpy copies for de-normalization
    y_std_np  = ds_train.y_std.astype("float32")
    y_mean_np = ds_train.y_mean.astype("float32")

    best_mse, best_ep = float("inf"), -1
    for ep in range(1, args.epochs+1):
        tr = train_one_epoch(model, tr_loader, opt, device)
        va_mse, va_rmse_cp = eval_epoch(model, va_loader, device, y_std_np, y_mean_np)
        sched.step()
        print(f"[ep {ep:02d}] train {tr:.5f} | val {va_mse:.5f} | val_RMSE_Cp {va_rmse_cp:.3f}")
        if va_mse < best_mse:
            best_mse, best_ep = va_mse, ep
            torch.save(model.state_dict(), os.path.join(args.outdir, "best.pt"))
    print(f"[done] best val MSE {best_mse:.5f} at ep {best_ep}")

    # Quick qualitative checks
    y_std = float(ds_train.y_std[0,0,0,0]); y_mean = float(ds_train.y_mean[0,0,0,0])
    for k in range(min(3, len(val_idx))):
        save_preview(model, ds_val, device, os.path.join(args.outdir, f"preview_{k}.png"),
                     idx=k, y_std=y_std, y_mean=y_mean)

if __name__ == "__main__":
    main()
