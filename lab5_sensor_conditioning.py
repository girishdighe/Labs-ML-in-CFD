# lab5_sensor_conditioning.py
# Sensor-conditioned reconstruction on AirfRANS rasters.
# Robust loader: supports .npz files, per-sample directories with .npy, and .h5 if h5py is present.

import argparse, os, glob, json, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Utilities
# -----------------------
def to_chw(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None, ...]
    assert arr.ndim == 3, f"expected (C,H,W), got {arr.shape}"
    return arr

def zscore_fit(Y):  # Y: [N,C,H,W]
    mean = Y.mean(axis=(0,2,3), keepdims=True)
    std = Y.std(axis=(0,2,3), keepdims=True) + 1e-8
    return mean, std

def zscore_apply(Y, mean, std):
    return (Y - mean) / std

def tv_loss(x):
    dx = x[..., 1:, :] - x[..., :-1, :]
    dy = x[..., :, 1:] - x[..., :, :-1]
    return dx.pow(2).mean() + dy.pow(2).mean()

def divergence(u, v):
    dudx = u[..., 1:, :] - u[..., :-1, :]
    dvdy = v[..., :, 1:] - v[..., :, :-1]
    dudx = F.pad(dudx, (0,0,1,0))
    dvdy = F.pad(dvdy, (1,0,0,0))
    return dudx + dvdy

def rmse(a, b):
    return float(torch.sqrt(F.mse_loss(a, b) + 1e-12))

def _resize_hw(x, H, W):
    x = np.asarray(x)
    if x.ndim == 2:
        t = torch.from_numpy(x)[None,None,:,:].float()
        t = F.interpolate(t, size=(H,W), mode="bilinear", align_corners=False)
        return t[0,0].cpu().numpy()
    elif x.ndim == 3:
        t = torch.from_numpy(x)[None,:,:,:].float()
        t = F.interpolate(t, size=(H,W), mode="bilinear", align_corners=False)
        return t[0].cpu().numpy()
    else:
        raise ValueError(f"bad ndim for resize: {x.shape}")

def _get(z, names, default=None):
    for n in names:
        if n in z: return z[n]
    if default is not None: return default
    raise KeyError(f"none of {names} in keys {list(z.keys())}")

# -----------------------
# Robust data discovery & loading
# -----------------------
def discover_items(root, split, glob_pat=None, limit=None):
    if glob_pat:
        items = sorted(glob.glob(glob_pat, recursive=True))
    else:
        # try common layouts automatically
        candidates = []
        # flat/recursive .npz
        for pat in [os.path.join(root, split, "*.npz"),
                    os.path.join(root, "**", split, "*.npz"),
                    os.path.join(root, "*.npz"),
                    os.path.join(root, "**", "*.npz")]:
            candidates += glob.glob(pat, recursive=True)
        # per-sample directories (contain p.npy/Cp.npy or u.npy etc.)
        for pat in [os.path.join(root, split, "*"),
                    os.path.join(root, "**", split, "*")]:
            for d in glob.glob(pat, recursive=True):
                if os.path.isdir(d):
                    fs = [f.lower() for f in os.listdir(d)]
                    if any(x in fs for x in ["p.npy","cp.npy","pressure.npy","u.npy","ux.npy","v.npy","vy.npy"]):
                        candidates.append(d)
        # h5/hdf5
        for pat in [os.path.join(root, split, "*.h5"),
                    os.path.join(root, split, "*.hdf5"),
                    os.path.join(root, "**", split, "*.h5"),
                    os.path.join(root, "**", split, "*.hdf5"),
                    os.path.join(root, "*.h5"),
                    os.path.join(root, "*.hdf5"),
                    os.path.join(root, "**", "*.h5"),
                    os.path.join(root, "**", "*.hdf5")]:
            candidates += glob.glob(pat, recursive=True)

        items = sorted(set(candidates))

    if not items:
        raise FileNotFoundError(
            f"No samples found under {root}. "
            "Pass --glob 'AirfRANS/train/**/*.npz' (or a directory pattern) if your layout is custom."
        )
    if limit: items = items[:limit]
    return items

def load_one_sample(item, var, H, W):
    """Return (sdf, aoa_map, re_map, target_chw) as numpy arrays."""
    # case 1: item is .npz
    if os.path.isfile(item) and item.lower().endswith(".npz"):
        with np.load(item) as z:
            sdf = _get(z, ["sdf","SDF","phi","Phi"])
            aoa = _get(z, ["aoa","alpha","AoA","Alpha"], 0.0)
            re  = _get(z, ["re","Re","Reynolds"], 1.0)

            if var == "p":
                p = None
                for name in ["p","P","Cp","cp","pressure","Pressure"]:
                    if name in z: p = z[name]; break
                if p is None: raise KeyError(f"{item}: pressure not found")
                tgt = _resize_hw(p, H, W)[None, ...]
            else:
                u = _get(z, ["u","U","Ux","u_x","uMean","velx"])
                v = _get(z, ["v","Vy","v_y","vMean","vely"])
                tgt = np.stack([_resize_hw(u,H,W), _resize_hw(v,H,W)], axis=0)

            sdf = _resize_hw(sdf, H, W)
            aoa_map = np.full((H,W), np.float32(aoa))
            re_map  = np.full((H,W),  np.float32(re))
            return sdf, aoa_map, re_map, tgt

    # case 2: item is directory with .npy and maybe meta
    if os.path.isdir(item):
        files = {f.lower(): os.path.join(item,f) for f in os.listdir(item)}
        # helpers to load if present
        def load_opt(keys, default=None):
            for k in keys:
                if k in files:
                    return np.load(files[k])
            return default
        def load_txt(keys, default=None):
            for k in keys:
                if k in files:
                    try:
                        return float(open(files[k]).read().strip())
                    except Exception:
                        pass
            return default

        sdf = load_opt(["sdf.npy","phi.npy"])
        if sdf is None:
            raise KeyError(f"{item}: missing sdf.npy/phi.npy")
        sdf = _resize_hw(sdf, H, W)

        aoa = load_txt(["aoa.txt","alpha.txt"], 0.0)
        re  = load_txt(["re.txt","reynolds.txt"], 1.0)
        aoa_map = np.full((H,W), np.float32(aoa))
        re_map  = np.full((H,W), np.float32(re))

        if var == "p":
            p = load_opt(["p.npy","cp.npy","pressure.npy"])
            if p is None: raise KeyError(f"{item}: missing p.npy/Cp.npy")
            tgt = _resize_hw(p, H, W)[None, ...]
        else:
            u = load_opt(["u.npy","ux.npy","u_x.npy"])
            v = load_opt(["v.npy","vy.npy","v_y.npy"])
            if u is None or v is None: raise KeyError(f"{item}: missing u.npy/v.npy")
            tgt = np.stack([_resize_hw(u,H,W), _resize_hw(v,H,W)], axis=0)

        return sdf, aoa_map, re_map, tgt

    # case 3: h5/hdf5
    if os.path.isfile(item) and item.lower().endswith((".h5",".hdf5")):
        try:
            import h5py
        except Exception as e:
            raise RuntimeError(f"h5py not available to read {item}: {e}")
        with h5py.File(item, "r") as h:
            keys = list(h.keys())
            def hget(names, default=None):
                for n in names:
                    if n in h: return h[n][()]
                if default is not None: return default
                raise KeyError(f"{item}: none of {names} in {keys}")

            sdf = hget(["sdf","SDF","phi","Phi"])
            aoa = hget(["aoa","alpha","AoA","Alpha"], 0.0)
            re  = hget(["re","Re","Reynolds"], 1.0)

            if var == "p":
                p = None
                for n in ["p","P","Cp","cp","pressure","Pressure"]:
                    if n in h: p = h[n][()]; break
                if p is None: raise KeyError(f"{item}: pressure not found")
                tgt = _resize_hw(p, H, W)[None, ...]
            else:
                u = hget(["u","U","Ux","u_x","uMean","velx"])
                v = hget(["v","Vy","v_y","vMean","vely"])
                tgt = np.stack([_resize_hw(u,H,W), _resize_hw(v,H,W)], axis=0)

            sdf = _resize_hw(sdf, H, W)
            aoa_map = np.full((H,W), np.float32(aoa))
            re_map  = np.full((H,W), np.float32(re))
            return sdf, aoa_map, re_map, tgt

    raise ValueError(f"Unrecognized sample path: {item}")

def load_dataset(root, var, M, H, W, split, glob_pat=None):
    items = discover_items(root, split, glob_pat, limit=M)
    X_sdf, X_aoa, X_re, Y = [], [], [], []
    for it in items:
        sdf, aoa_map, re_map, tgt = load_one_sample(it, var, H, W)
        X_sdf.append(sdf[None,...])
        X_aoa.append(aoa_map[None,...])
        X_re.append(re_map[None,...])
        Y.append(tgt)
    X_sdf = np.stack(X_sdf, axis=0).astype(np.float32)
    X_aoa = np.stack(X_aoa, axis=0).astype(np.float32)
    X_re  = np.stack(X_re,  axis=0).astype(np.float32)
    Y     = np.stack(Y,     axis=0).astype(np.float32)
    return X_sdf, X_aoa, X_re, Y

# -----------------------
# Sensor features
# -----------------------
def sample_sensors(H, W, K):
    ys = np.random.randint(0, H, size=K)
    xs = np.random.randint(0, W, size=K)
    return ys, xs

def gaussian_splats(ys, xs, H, W, sigma=2.0):
    yy, xx = np.mgrid[0:H, 0:W]
    spl = np.zeros((H, W), np.float32)
    for y,x in zip(ys,xs):
        g = np.exp(-((yy-y)**2 + (xx-x)**2) / (2*sigma**2))
        spl += g
    if spl.max() > 0: spl /= (spl.max() + 1e-8)
    return spl

def idw_prior(ys, xs, vals, H, W, power=2.0, eps=1e-6):
    yy, xx = np.mgrid[0:H, 0:W]
    num, den = np.zeros((H,W),np.float32), np.zeros((H,W),np.float32)
    for y,x,v in zip(ys,xs,vals):
        d2 = (yy-y)**2 + (xx-x)**2
        w = 1.0 / (np.power(np.sqrt(d2 + eps), power) + eps)
        num += w * v
        den += w
    return num / (den + 1e-8)

# -----------------------
# Dataset
# -----------------------
class SensorCondDS(Dataset):
    def __init__(self, root, var, M, H, W, K, split="train", sigma=2.0, include_idw=True, glob_pat=None):
        self.var, self.K, self.sigma, self.include_idw = var, K, sigma, include_idw
        X_sdf, X_aoa, X_re, Y = load_dataset(root, var, M, H, W, split, glob_pat)
        self.X_sdf, self.X_aoa, self.X_re = X_sdf, X_aoa, X_re
        self.Y = Y
        self.N, self.C, self.H, self.W = Y.shape

        yy, xx = np.mgrid[0:H, 0:W]
        self.xx = (xx/(W-1)).astype(np.float32)[None, ...]
        self.yy = (yy/(H-1)).astype(np.float32)[None, ...]
        self.mean, self.std = zscore_fit(self.Y)

    def __len__(self): return self.N

    def __getitem__(self, i):
        sdf, aoa, re = self.X_sdf[i], self.X_aoa[i], self.X_re[i]
        Yabs = self.Y[i]                      # (C,H,W)
        Yz   = (Yabs - self.mean) / self.std  # z-score

        ys, xs = sample_sensors(self.H, self.W, self.K)

        mask = np.zeros((1, self.H, self.W), np.float32)
        vals = np.zeros((self.C, self.H, self.W), np.float32)
        svals = Yz[:, ys, xs]   # (C,K)
        for k in range(self.K):
            y, x = ys[k], xs[k]
            mask[0, y, x] = 1.0
            vals[:, y, x] = svals[:, k]

        spl = gaussian_splats(ys, xs, self.H, self.W, sigma=self.sigma)[None, ...]
        if self.include_idw:
            idw = np.concatenate(
                [idw_prior(ys, xs, svals[c], self.H, self.W)[None, ...] for c in range(self.C)],
                axis=0
            )
        else:
            idw = np.zeros_like(vals)

        X = np.concatenate([self.xx, self.yy, sdf, aoa, re, mask, spl, vals, idw], axis=0)
        return (
            torch.from_numpy(X).float(),
            torch.from_numpy(Yz).float(),
            torch.from_numpy(Yabs).float(),
            torch.from_numpy(self.mean.squeeze()).float(),
            torch.from_numpy(self.std.squeeze()).float(),
            torch.from_numpy(mask).float(),
            torch.from_numpy(svals).float(),
            torch.from_numpy(np.stack([ys, xs], 1)).long()
        )

# -----------------------
# Model
# -----------------------
class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, k=3, d=1):
        super().__init__()
        p = ((k-1)//2)*d
        self.op = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, padding=p, dilation=d, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.op(x)

class UNetSmall(nn.Module):
    def __init__(self, c_in, c_out, width=64):
        super().__init__()
        w = width
        self.stem = nn.Sequential(ConvBNReLU(c_in, w//2, 3, 1),
                                  ConvBNReLU(w//2, w, 3, 1))
        self.enc1 = nn.Sequential(nn.MaxPool2d(2), ConvBNReLU(w,   2*w, 3, 1))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), ConvBNReLU(2*w, 4*w, 3, 2))
        self.mid  = ConvBNReLU(4*w, 4*w, 3, 4)
        self.up2  = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                  ConvBNReLU(4*w, 2*w, 3, 1))
        self.up1  = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                  ConvBNReLU(3*w, w, 3, 1))
        self.head = nn.Conv2d(2*w, c_out, 3, padding=1)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        xm = self.mid(x2)
        y2 = self.up2(xm)
        y1 = self.up1(torch.cat([y2, x1], dim=1))
        out = self.head(torch.cat([y1, x0], dim=1))
        return out

# -----------------------
# DataLoaders
# -----------------------
def make_loaders(root, var, M, H, W, K, include_idw, batch, glob_pat):
    ds = SensorCondDS(root, var, M, H, W, K, split="train", include_idw=include_idw, glob_pat=glob_pat)
    n_val = max(int(0.15*len(ds)), 16)
    n_tr  = len(ds) - n_val
    tr_ds, va_ds = torch.utils.data.random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(42))
    tr = DataLoader(tr_ds, batch_size=batch, shuffle=True,  num_workers=0, drop_last=True)
    va = DataLoader(va_ds, batch_size=batch, shuffle=False, num_workers=0)
    return tr, va

# -----------------------
# Training
# -----------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va = make_loaders(args.root, args.var, args.M, args.H, args.W, args.K,
                          include_idw=args.include_idw, batch=args.batch, glob_pat=args.glob)

    # infer channels
    with torch.no_grad():
        Xb, Yzb, *_ = next(iter(tr))
    c_in, c_out = Xb.shape[1], Yzb.shape[1]
    model = UNetSmall(c_in, c_out, width=args.width).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)

    def loss_fn(pred_z, tgt_z, mask, svals, idxs, mean, std, varname):
        loss = F.mse_loss(pred_z, tgt_z)

        # sensor-consistency (pred at sensor pixels equals sensor values)
        if args.lambda_sensor > 0:
            B = pred_z.shape[0]
            sens_pred, sens_true = [], []
            for b in range(B):
                for k in range(idxs[b].shape[0]):
                    y, x = idxs[b][k]
                    sens_pred.append(pred_z[b, :, y, x])
                    sens_true.append(svals[b][:, k])
            if sens_pred:
                sens_pred = torch.stack(sens_pred)
                sens_true = torch.stack(sens_true)
                loss = loss + args.lambda_sensor * F.mse_loss(sens_pred, sens_true)

        # TV smoothing on absolute field
        pred_abs = pred_z * std[None,:,None,None] + mean[None,:,None,None]
        loss = loss + args.lambda_tv * tv_loss(pred_abs)

        # divergence penalty for uv
        if varname == "uv" and pred_abs.shape[1] == 2 and args.lambda_div > 0:
            div = divergence(pred_abs[:,0:1], pred_abs[:,1:2])
            loss = loss + args.lambda_div * (div.pow(2).mean())
        return loss

    best = (1e9, -1)
    for ep in range(1, args.epochs+1):
        model.train()
        tr_running = 0.0
        for X, Yz, Yabs, mean, std, mask, svals, idxs in tr:
            X, Yz = X.to(device), Yz.to(device)
            mean, std = mean.to(device), std.to(device)
            svals, idxs = svals.to(device), idxs.to(device)

            pred_z = model(X)
            if args.hard_clamp:
                # enforce exact sensor values in z-space
                for b in range(pred_z.shape[0]):
                    for k in range(idxs[b].shape[0]):
                        y, x = idxs[b][k]
                        pred_z[b, :, y, x] = svals[b][:, k]

            loss = loss_fn(pred_z, Yz, mask, svals, idxs, mean, std, args.var)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_running += loss.item()

        # validation
        model.eval()
        with torch.no_grad():
            v_mse, v_abs = 0.0, 0.0; n=0
            for X, Yz, Yabs, mean, std, *_ in va:
                X, Yz, Yabs = X.to(device), Yz.to(device), Yabs.to(device)
                mean, std = mean.to(device), std.to(device)
                pred_z = model(X)
                pred_abs = pred_z * std[None,:,None,None] + mean[None,:,None,None]
                v_mse += F.mse_loss(pred_z, Yz).item()
                v_abs += rmse(pred_abs, Yabs)
                n += 1
            v_mse /= n; v_abs /= n
        print(f"[ep {ep:02d}] train {tr_running/len(tr):.5f} | val_mse(z) {v_mse:.5f} | val_RMSE_abs {v_abs:.3f}")
        if v_mse < best[0]:
            best = (v_mse, ep)
    print(f"[done] best val_mse(z) {best[0]:.5f} @ ep {best[1]}")

# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--var", type=str, choices=["p","uv"], default="p")
    ap.add_argument("--M", type=int, default=200)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--W", type=int, default=64)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--include_idw", action="store_true")
    ap.add_argument("--lambda_sensor", type=float, default=20.0)
    ap.add_argument("--lambda_tv", type=float, default=1e-4)
    ap.add_argument("--lambda_div", type=float, default=0.0)
    ap.add_argument("--hard_clamp", action="store_true")
    ap.add_argument("--glob", type=str, default=None,
                   help="Optional glob pattern to find samples if auto-discovery fails "
                        "(e.g., 'AirfRANS/train/**/*.npz' or 'AirfRANS/train/*').")
    args = ap.parse_args()
    train(args)

if __name__ == "__main__":
    main()
