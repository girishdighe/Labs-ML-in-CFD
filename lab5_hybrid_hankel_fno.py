# lab5_hybrid_hankel_fno.py
# Hybrid = Hankel-DMD core + tiny FNO residual (CPU friendly)

import os, argparse, math
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# --------------------------- Utilities ---------------------------

def rmse(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b)**2)))

def shared_limits(a, b):
    both = np.concatenate([np.real(a).ravel(), np.real(b).ravel()])
    vmin = np.percentile(both, 1); vmax = np.percentile(both, 99)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(both.min()), float(both.max())
    return vmin, vmax

def imsave_pair(left, right, path, tl="True", tr="Pred"):
    vmin, vmax = shared_limits(left, right)
    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1); plt.imshow(left, origin="lower", vmin=vmin, vmax=vmax); plt.title(tl); plt.colorbar()
    plt.subplot(1,2,2); plt.imshow(right, origin="lower", vmin=vmin, vmax=vmax); plt.title(tr); plt.colorbar()
    plt.tight_layout(); plt.savefig(path); plt.close()

# -------------------- AirfRANS loading & rasters -----------------

def ensure_airfrans(root):
    import airfrans as af
    dataset_dir = os.path.join(root, "Dataset")
    if not os.path.exists(dataset_dir):
        print("[setup] Downloading AirfRANS (preprocessed, no OpenFOAM files)…")
        af.dataset.download(root=root, file_name="Dataset", unzip=True, OpenFOAM=False)
    return dataset_dir

def load_airfrans_as_rasters(root, var="p", H=128, W=128, M=80):
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
    return out  # keys: p,u,v,Xg,Yg

# ------------------- POD ordering (pseudo-time) ------------------

def center_and_svd_rows(X_flat):
    mean = X_flat.mean(axis=0, keepdims=True)
    Xc = X_flat - mean
    U, S, VT = np.linalg.svd(Xc, full_matrices=False)
    return U, S, VT, mean

def build_sequence_by_pod(X, blend=0.8):
    N,H,W = X.shape
    X_flat = X.reshape(N, H*W)
    U, S, VT, mean = center_and_svd_rows(X_flat)
    if U.shape[1] == 1:
        scores = U[:,0]*S[0]
    else:
        scores = blend*(U[:,0]*S[0]) + (1.0-blend)*(U[:,1]*S[1])
    order = np.argsort(scores)
    return X[order], order

# ---------------------- Hankel-DMD core (Lab 4A) -----------------

def build_hankel_pairs(X_cols0, q=6, stride=1):
    Fdim, T = X_cols0.shape
    L = T - (q-1)*stride
    if L < 2: raise ValueError("Not enough snapshots for q.")
    K = L - 1
    qF = q*Fdim
    Z0 = np.zeros((qF, K), dtype=X_cols0.dtype)
    Z1 = np.zeros((qF, K), dtype=X_cols0.dtype)
    for k in range(K):
        cols0 = [k + j*stride for j in range(q)]
        cols1 = [k+1 + j*stride for j in range(q)]
        Z0[:, k] = np.concatenate([X_cols0[:, c] for c in cols0], axis=0)
        Z1[:, k] = np.concatenate([X_cols0[:, c] for c in cols1], axis=0)
    start = (T - q*stride)
    z_init = np.concatenate([X_cols0[:, start + j*stride] for j in range(q)], axis=0)
    return Z0, Z1, z_init

def fit_hankel_dmd(X_cols, q=6, stride=1, energy=0.99, r=None, ridge=1e-3):
    Z0, Z1, z_init = build_hankel_pairs(X_cols, q=q, stride=stride)
    U, S, VT = np.linalg.svd(Z0, full_matrices=False)
    if r is None:
        e=S**2; cum=np.cumsum(e)/np.sum(e); r=int(np.searchsorted(cum, energy)+1)
    Ur = U[:,:r]; Sr = S[:r]; Vr = VT[:r,:].T
    Sr_inv = np.diag(1.0/Sr)

    # Ridge in reduced coords
    Xr = Ur.T @ Z0
    Yr = Ur.T @ Z1
    Atil = Yr @ Xr.T @ np.linalg.inv(Xr @ Xr.T + ridge*np.eye(r))

    eigvals, Weig = np.linalg.eig(Atil)
    Phi = (Z1 @ Vr) @ Sr_inv @ Weig

    y0 = Ur.T @ z_init
    b0 = np.linalg.solve(Weig, y0)

    return {"Phi": Phi, "eigvals": eigvals, "Ur": Ur, "Weig": Weig,
            "r": r, "q": q, "Fdim": X_cols.shape[0], "z_init": z_init,
            "ridge": ridge, "Vr": Vr, "Sr_inv": Sr_inv}

def forecast_hankel_dmd(model, steps):
    Phi, lam, Ur, Weig = model["Phi"], model["eigvals"], model["Ur"], model["Weig"]
    z_init = model["z_init"]; Fdim = model["Fdim"]; q = model["q"]
    y_init = Ur.T @ z_init
    b_init = np.linalg.solve(Weig, y_init)
    preds_lastblock = []
    for k in range(1, steps+1):
        coeff = (lam**k) * b_init
        z_k = np.real(Phi @ coeff)
        last = z_k[(q-1)*Fdim : q*Fdim]
        preds_lastblock.append(last)
    return np.stack(preds_lastblock, axis=1)  # [Fdim, steps]

def hankel_onestep_from_window(model, X_cols0, start_col):
    """Predict x_{t+q} from centered window [t ... t+q-1]."""
    Fdim = model["Fdim"]; q = model["q"]; Ur = model["Ur"]; Weig = model["Weig"]; Phi = model["Phi"]; lam = model["eigvals"]
    z = np.concatenate([X_cols0[:, start_col + j] for j in range(q)], axis=0)  # [qF]
    y = Ur.T @ z
    b = np.linalg.solve(Weig, y)
    z1 = np.real(Phi @ b)                                  # one step
    return z1[(q-1)*Fdim : q*Fdim]                         # last block

# ---------------------- Tiny FNO residual learner ----------------

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.m1, self.m2 = modes1, modes2
        self.wr = nn.Parameter(torch.randn(in_channels, out_channels, modes1, modes2)*0.02)
        self.wi = nn.Parameter(torch.randn(in_channels, out_channels, modes1, modes2)*0.02)

    def cmul(self, a, br, bi):
        ar, ai = a.real, a.imag
        real = torch.einsum("bixy,ioxy->boxy", ar, br) - torch.einsum("bixy,ioxy->boxy", ai, bi)
        imag = torch.einsum("bixy,ioxy->boxy", ar, bi) + torch.einsum("bixy,ioxy->boxy", ai, br)
        return torch.complex(real, imag)

    def forward(self, x):
        B,C,H,W = x.shape
        xft = torch.fft.rfft2(x, norm="ortho")
        out = torch.zeros(B, self.wr.shape[1], H, W//2+1, dtype=torch.complex64, device=x.device)
        m1, m2 = min(self.m1, H), min(self.m2, W//2+1)
        out[:,:, :m1, :m2] = self.cmul(xft[:, :, :m1, :m2], self.wr[:, :, :m1, :m2], self.wi[:, :, :m1, :m2])
        return torch.fft.irfft2(out, s=(H,W), norm="ortho")

class FNO2dResidual(nn.Module):
    """Inputs: [x_norm, y_norm, baseline_pred] -> residual correction."""
    def __init__(self, in_ch=3, out_ch=1, width=32, modes=12, layers=4):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.spec = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(layers)])
        self.lin  = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.head = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_ch, 1))

    def forward(self, x):
        x = self.lift(x)
        for sc, ln in zip(self.spec, self.lin):
            x = F.gelu(sc(x) + ln(x))
        return self.head(x)

# ------------------- Residual dataset from Hankel -----------------

class ResidualWindows(Dataset):
    """
    Builds (inp, tgt) pairs from training sequence. For each window [t..t+q-1]:
    inp  = [x_norm, y_norm, baseline_pred_{t+q}]  (z-scored per train)
    tgt  = true x_{t+q}                            (z-scored per train)
    """
    def __init__(self, X_train0_cols, model, H, W, xgrid, ygrid, q=6,
                 idx_range=None, stats=None):
        self.X = X_train0_cols
        self.model = model
        self.H, self.W = H, W
        self.q = q
        self.xn = (xgrid - xgrid.min())/(xgrid.max()-xgrid.min())*2 - 1
        self.yn = (ygrid - ygrid.min())/(ygrid.max()-ygrid.min())*2 - 1

        Ttr = X_train0_cols.shape[1]
        # Valid starting indices: 0 .. Ttr - q - 1 (so target at t+q exists)
        self.idxs = np.arange(0, Ttr - q) if idx_range is None else idx_range

        # Build full tensors once
        inp_list, tgt_list = [], []
        for t in self.idxs:
            base = hankel_onestep_from_window(model, X_train0_cols, t)        # [Fdim]
            tgt  = X_train0_cols[:, t + q]                                    # [Fdim]
            inp = np.stack([self.xn, self.yn, base.reshape(H, W)], axis=0)    # [3,H,W]
            inp_list.append(inp.astype("float32"))
            tgt_list.append(tgt.reshape(1, H, W).astype("float32"))
        self.inp = np.stack(inp_list, axis=0)  # [N,3,H,W]
        self.tgt = np.stack(tgt_list, axis=0)  # [N,1,H,W]

        # Stats (train only)
        if stats is None:
            self.x_mean = self.inp.mean(axis=(0,2,3), keepdims=True)
            self.x_std  = self.inp.std(axis=(0,2,3), keepdims=True) + 1e-6
            self.y_mean = self.tgt.mean(axis=(0,2,3), keepdims=True)
            self.y_std  = self.tgt.std(axis=(0,2,3), keepdims=True) + 1e-6
        else:
            self.x_mean, self.x_std, self.y_mean, self.y_std = stats

        self.inp = (self.inp - self.x_mean)/self.x_std
        self.tgt = (self.tgt - self.y_mean)/self.y_std

    def __len__(self): return self.inp.shape[0]
    def __getitem__(self, i):
        return torch.from_numpy(self.inp[i]), torch.from_numpy(self.tgt[i])

# ------------------------------ Main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="AirfRANS")
    ap.add_argument("--var", type=str, default="p", choices=["p"])
    ap.add_argument("--M", type=int, default=80)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--W", type=int, default=128)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--blend", type=float, default=0.8)

    # Hankel-DMD
    ap.add_argument("--q", type=int, default=6)
    ap.add_argument("--rank", type=int, default=12)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--energy", type=float, default=0.99)

    # FNO residual
    ap.add_argument("--fno_width", type=int, default=32)
    ap.add_argument("--fno_modes", type=int, default=12)
    ap.add_argument("--fno_layers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="lab5_outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cpu")

    # 1) Load rasters of p
    bundle = load_airfrans_as_rasters(args.root, var=args.var, H=args.H, W=args.W, M=args.M)
    P = bundle["p"]   # [N,H,W]
    Xg, Yg = bundle["Xg"], bundle["Yg"]
    N, H, W = P.shape
    Fdim = H*W

    # 2) Pseudo-time ordering & column form
    P_ord, order = build_sequence_by_pod(P, blend=args.blend)
    P_cols = P_ord.reshape(N, Fdim).T  # [Fdim,T]
    T = P_cols.shape[1]

    # 3) Split and center by train mean
    Ttr = max(12, int(np.floor(args.train_frac * T)))
    P_tr = P_cols[:, :Ttr]
    P_te = P_cols[:, Ttr:]
    mean_tr = P_tr.mean(axis=1, keepdims=True)
    P_tr0 = P_tr - mean_tr
    P_te0 = P_te - mean_tr
    Tte = P_te0.shape[1]
    print(f"[split] T={T}, train={Ttr}, test={Tte}")

    # 4) Fit Hankel-DMD on train
    model = fit_hankel_dmd(P_tr0, q=args.q, energy=args.energy, r=args.rank, ridge=args.ridge)
    print(f"[hankel] q={args.q}, r={model['r']}, ridge={args.ridge:g}")

    # Diagnostics: eigenvalues
    eig = model["eigvals"]
    th = np.linspace(0, 2*np.pi, 400)
    plt.figure()
    plt.plot(np.cos(th), np.sin(th), '--', linewidth=1)
    plt.scatter(eig.real, eig.imag, s=20)
    plt.axis('equal'); plt.xlabel("Re(λ)"); plt.ylabel("Im(λ)")
    plt.title("Hankel-DMD eigenvalues (discrete)")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "hankel_eigs.png")); plt.close()

    # 5) Build residual dataset from train windows
    total_windows = P_tr0.shape[1] - args.q
    split_w = max(4, int(np.floor(0.8 * total_windows)))
    idx_train = np.arange(0, split_w)
    idx_val   = np.arange(split_w, total_windows)

    ds_res_tr = ResidualWindows(P_tr0, model, H, W, Xg, Yg, q=args.q, idx_range=idx_train)
    stats = (ds_res_tr.x_mean, ds_res_tr.x_std, ds_res_tr.y_mean, ds_res_tr.y_std)
    ds_res_va = ResidualWindows(P_tr0, model, H, W, Xg, Yg, q=args.q, idx_range=idx_val, stats=stats)

    tr_loader = DataLoader(ds_res_tr, batch_size=args.batch, shuffle=True, num_workers=0)
    va_loader = DataLoader(ds_res_va, batch_size=args.batch, shuffle=False, num_workers=0)

    # 6) Residual FNO
    fno = FNO2dResidual(in_ch=3, out_ch=1, width=args.fno_width, modes=args.fno_modes, layers=args.fno_layers).to(device)
    opt = torch.optim.AdamW(fno.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def crit(pred, y):  # robust loss (uses torch.nn.functional as F)
        return F.smooth_l1_loss(pred, y, beta=0.02) + 0.1*F.mse_loss(pred, y)

    print(f"[residual] params ~ {sum(p.numel() for p in fno.parameters())/1e6:.2f}M")
    best, best_ep = float("inf"), -1
    for ep in range(1, args.epochs+1):
        # train
        fno.train(); acc=0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = fno(xb)
            loss = crit(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            acc += loss.item()*xb.size(0)
        tr_loss = acc/len(ds_res_tr)

        # val
        fno.eval(); acc=0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = fno(xb)
                loss = F.mse_loss(pred, yb)
                acc += loss.item()*xb.size(0)
        va_mse = acc/len(ds_res_va)
        sched.step()
        print(f"[ep {ep:02d}] residual train {tr_loss:.5f} | val_mse {va_mse:.5f}")
        if va_mse < best:
            best, best_ep = va_mse, ep
            torch.save(fno.state_dict(), os.path.join(args.outdir, "fno_residual_best.pt"))
    print(f"[residual] best val_mse {best:.5f} @ ep {best_ep}")

    # 7) Test: baseline Hankel vs Hybrid (Hankel + residual)
    if Tte > 0:
        # Baseline Hankel rollout (from last train window)
        base_preds = forecast_hankel_dmd(model, steps=Tte)                      # [Fdim,Tte]
        base_imgs  = base_preds + mean_tr                                       # de-center

        # Hybrid: add residual per step (CNN sees normalized [x,y, base] channels)
        x_mean = ds_res_tr.x_mean; x_std = ds_res_tr.x_std
        y_mean = ds_res_tr.y_mean; y_std = ds_res_tr.y_std

        xn = (Xg - Xg.min())/(Xg.max()-Xg.min())*2 - 1
        yn = (Yg - Yg.min())/(Yg.max()-Yg.min())*2 - 1

        fno.eval()
        hybrid_imgs = []
        for j in range(Tte):
            base = base_preds[:, j].reshape(H, W) + 0.0      # centered baseline
            inp = np.stack([xn, yn, base], axis=0)[None].astype("float32")      # [1,3,H,W]
            inp_z = (inp - x_mean)/x_std
            with torch.no_grad():
                corr_z = fno(torch.from_numpy(inp_z)).cpu().numpy()   # [1,1,H,W]
            corr = corr_z * y_std + y_mean                            # back to p units (centered residual)
            hybrid = base + corr[0,0]                                  # still centered
            hybrid_imgs.append((hybrid + mean_tr.reshape(H,W)))        # de-center
        hybrid_imgs = np.stack(hybrid_imgs, axis=2)                    # [H,W,Tte]

        # Ground truth test images (physical)
        gt_imgs = (P_te0 + mean_tr).reshape(H, W, Tte)

        # Metrics
        base_rmse = [rmse(base_imgs[:,j].reshape(H,W), gt_imgs[:,:,j]) for j in range(Tte)]
        hybr_rmse = [rmse(hybrid_imgs[:,:,j],           gt_imgs[:,:,j]) for j in range(Tte)]
        copy_rmse = [rmse((P_tr[:,-1] + mean_tr[:,0]).reshape(H,W), gt_imgs[:,:,j]) for j in range(Tte)]
        print(f"[test] mean RMSE over {Tte} steps: Hankel={np.mean(base_rmse):.3f} | Hybrid={np.mean(hybr_rmse):.3f} | copy-last={np.mean(copy_rmse):.3f}")

        # Save a couple of frames
        for j in [0, min(1, Tte-1)]:
            imsave_pair(gt_imgs[:,:,j], base_imgs[:,j].reshape(H,W),
                        os.path.join(args.outdir, f"pair_baseline_{j}.png"),
                        tl=f"True[t+{j+1}]", tr=f"Hankel-DMD[t+{j+1}]")
            imsave_pair(gt_imgs[:,:,j], hybrid_imgs[:,:,j],
                        os.path.join(args.outdir, f"pair_hybrid_{j}.png"),
                        tl=f"True[t+{j+1}]", tr=f"Hybrid[t+{j+1}]")

        # Plot per-step RMSE
        k = np.arange(1, Tte+1)
        plt.figure()
        plt.plot(k, base_rmse, "-o", label="Hankel")
        plt.plot(k, hybr_rmse, "-o", label="Hybrid")
        plt.plot(k, copy_rmse, "-o", label="Copy-last")
        plt.xlabel("Test step"); plt.ylabel("RMSE (p/ρ units)")
        plt.title("Per-step RMSE")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "rmse_per_step.png")); plt.close()
    else:
        print("[test] No test horizon.")

    print(f"[done] outputs in: {args.outdir}")

if __name__ == "__main__":
    main()
