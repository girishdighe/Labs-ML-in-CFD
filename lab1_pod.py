# lab1_airfrans_pod.py
import argparse, os
import numpy as np
import matplotlib.pyplot as plt

def ensure_airfrans(root):
    import airfrans as af
    dataset_dir = os.path.join(root, "Dataset")
    if not os.path.exists(dataset_dir):
        print("[setup] Downloading AirfRANS (preprocessed, no OpenFOAM files)…")
        af.dataset.download(root=root, file_name="Dataset", unzip=True, OpenFOAM=False)
    return dataset_dir

def load_airfrans_as_rasters(root, var="p", M=50, H=128, W=128):
    """
    var:
      'p'    -> p_over_rho  (index 9)
      'umag' -> sqrt(u^2+v^2) using indices 7,8
    """
    import airfrans as af
    from scipy.interpolate import griddata

    dataset_dir = ensure_airfrans(root)
    data_list, names = af.dataset.load(root=dataset_dir, task="full", train=True)
    M = min(M, len(data_list))
    print(f"[data] Using {M} samples from AirfRANS train split")

    # AirfRANS cropped domain: x in [-2,4], y in [-1.5,1.5]
    xg = np.linspace(-2.0, 4.0, W)
    yg = np.linspace(-1.5, 1.5, H)
    Xg, Yg = np.meshgrid(xg, yg)

    rasters = []
    for i in range(M):
        sim = data_list[i]  # shape: (N_pts, 12)
        x, y = sim[:, 0], sim[:, 1]

        if var == "p":
            values = sim[:, 9]  # p_over_rho: targets indices [7:u, 8:v, 9:p/rho, 10:nut]
        elif var == "umag":
            u, v = sim[:, 7], sim[:, 8]
            values = np.hypot(u, v)
        else:
            raise ValueError("var must be 'p' or 'umag'")

        Zi = griddata(points=np.c_[x, y], values=values, xi=(Xg, Yg), method="linear")
        # fill holes by nearest
        mask = np.isnan(Zi)
        if mask.any():
            Zi[mask] = griddata(np.c_[x, y], values, (Xg[mask], Yg[mask]), method="nearest")

        rasters.append(Zi.astype("float32"))

    arr = np.stack(rasters, axis=0)  # [N,H,W]
    return arr  # no file writing; returned in-memory

def pod_svd(X_flat, r=None, energy_threshold=None):
    mean = X_flat.mean(axis=0, keepdims=True)
    Xc = X_flat - mean
    U, S, VT = np.linalg.svd(Xc, full_matrices=False)

    if energy_threshold is not None and r is None:
        energy = (S**2)
        cum = np.cumsum(energy) / np.sum(energy)
        r = int(np.searchsorted(cum, energy_threshold) + 1)
    elif r is None:
        r = min(20, S.shape[0])

    Ur = U[:, :r]
    Sr = S[:r]
    VTr = VT[:r, :]
    Xc_rec = (Ur * Sr) @ VTr
    X_rec = Xc_rec + mean

    info = {
        "r": r,
        "singular_values": S,
        "energy_cum": np.cumsum(S**2)/np.sum(S**2),
        "mean": mean,
        "VT": VT
    }
    return X_rec, info

def rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))

def visualize_modes(VT, H, W, C, k_list=(0,1,2), outdir="pod_outputs"):
    os.makedirs(outdir, exist_ok=True)
    for k in k_list:
        mode_flat = VT[k, :]
        mode = mode_flat.reshape(H, W, C)
        img = mode[..., 0]
        plt.figure()
        plt.imshow(img, origin="lower")
        plt.title(f"POD Mode {k}")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"mode_{k}.png"))
        plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="AirfRANS", help="Where to download/cache AirfRANS")
    p.add_argument("--var", type=str, default="p", choices=["p","umag"], help="Field to rasterize")
    p.add_argument("--M", type=int, default=50, help="How many samples to use")
    p.add_argument("--H", type=int, default=128)
    p.add_argument("--W", type=int, default=128)
    p.add_argument("--energy", type=float, default=0.99, help="Cumulative energy threshold (0-1)")
    p.add_argument("--rank", type=int, default=None, help="Override rank r; if set, ignores --energy")
    p.add_argument("--outdir", type=str, default="pod_outputs")
    args = p.parse_args()

    # 1) Load AirfRANS → rasters
    X = load_airfrans_as_rasters(root=args.root, var=args.var, M=args.M, H=args.H, W=args.W)  # [N,H,W]
    N,H,W = X.shape
    X = X[:, :, :, None]          # [N,H,W,1]
    X_flat = X.reshape(N, H*W*1)  # [N,F]

    # 2) POD/SVD
    Xrec_flat, info = pod_svd(
        X_flat,
        r=args.rank,
        energy_threshold=None if args.rank is not None else args.energy
    )
    r = info["r"]; VT = info["VT"]; energy_cum = info["energy_cum"]; S = info["singular_values"]

    # 3) Metrics + plots
    Xrec = Xrec_flat.reshape(N, H, W, 1)
    err = rmse(Xrec, X)
    print(f"[POD] r = {r}")
    print(f"[POD] RMSE = {err:.6e}")
    print(f"[POD] Energy@r = {energy_cum[r-1]*100:.2f}%")

    os.makedirs(args.outdir, exist_ok=True)
    k = np.arange(1, len(S)+1)
    plt.figure(); plt.plot(k, S, marker='o'); plt.xlabel("Mode"); plt.ylabel("Singular value"); plt.title("Scree"); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "scree.png")); plt.close()

    plt.figure(); plt.plot(k, energy_cum, marker='o'); plt.axhline(args.energy, linestyle='--')
    plt.xlabel("Mode"); plt.ylabel("Cumulative energy"); plt.title("Energy curve"); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "energy_cumulative.png")); plt.close()

    visualize_modes(VT, H, W, 1, k_list=(0,1,2), outdir=args.outdir)

    for i in [0, min(1, N-1)]:
        plt.figure(figsize=(8,3))
        plt.subplot(1,2,1); plt.imshow(X[i,...,0], origin="lower"); plt.title(f"Original[{i}]"); plt.colorbar()
        plt.subplot(1,2,2); plt.imshow(Xrec[i,...,0], origin="lower"); plt.title(f"Recon (r={r})"); plt.colorbar()
        plt.tight_layout(); plt.savefig(os.path.join(args.outdir, f"recon_pair_{i}.png")); plt.close()

if __name__ == "__main__":
    main()
