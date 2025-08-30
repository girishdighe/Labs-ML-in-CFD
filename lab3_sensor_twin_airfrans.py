# lab3_sensor_twin_airfrans.py
# Sensor-conditioned POD reconstruction + DMD forecast nudging on AirfRANS.
# - One file, no data generation.
# - Reuses rasterization (Lab 1) and DMD core (Lab 2).
# - Does two tasks on a held-out test frame:
#   (A) Reconstruct from K sensors using POD only.
#   (B) Nudge a DMD forecast with the same sensors.

import argparse, os
import numpy as np
import matplotlib.pyplot as plt

# ------------------------- AirfRANS loading & rasterization -------------------------
def ensure_airfrans(root):
    import airfrans as af
    dataset_dir = os.path.join(root, "Dataset")
    if not os.path.exists(dataset_dir):
        print("[setup] Downloading AirfRANS (preprocessed, no OpenFOAM files)…")
        af.dataset.download(root=root, file_name="Dataset", unzip=True, OpenFOAM=False)
    return dataset_dir

def load_airfrans_as_rasters(root, var="p", M=60, H=128, W=128):
    import airfrans as af
    from scipy.interpolate import griddata

    dataset_dir = ensure_airfrans(root)
    data_list, names = af.dataset.load(root=dataset_dir, task="full", train=True)
    M = min(M, len(data_list))
    print(f"[data] Using {M} snapshots from AirfRANS train split")

    xg = np.linspace(-2.0, 4.0, W)
    yg = np.linspace(-1.5, 1.5, H)
    Xg, Yg = np.meshgrid(xg, yg)

    rasters = []
    for i in range(M):
        sim = data_list[i]  # (N_pts, 12)
        x, y = sim[:, 0], sim[:, 1]
        if var == "p":
            values = sim[:, 9]
        elif var == "umag":
            u, v = sim[:, 7], sim[:, 8]
            values = np.hypot(u, v)
        else:
            raise ValueError("var must be 'p' or 'umag'")

        Zi = griddata(points=np.c_[x, y], values=values, xi=(Xg, Yg), method="linear")
        mask = np.isnan(Zi)
        if mask.any():
            Zi[mask] = griddata(np.c_[x, y], values, (Xg[mask], Yg[mask]), method="nearest")
        rasters.append(Zi.astype("float32"))

    return np.stack(rasters, axis=0)  # [N,H,W]

# ------------------------- POD basis from train snapshots ---------------------------
def pod_from_train(X_train_cols, r=None, energy=0.99):
    """
    X_train_cols: [F, T_train] (columns are snapshots). Build POD on rows = snapshots.
    Returns: mean (F,), Phi (F,r), S (singular values), cum_energy.
    """
    F, Ttr = X_train_cols.shape
    X_rows = X_train_cols.T  # [Ttr, F]
    mean = X_rows.mean(axis=0, keepdims=True)  # [1,F]
    Xc = X_rows - mean
    U, S, VT = np.linalg.svd(Xc, full_matrices=False)  # Xc = U S VT
    if r is None:
        e = S**2
        cum = np.cumsum(e)/np.sum(e)
        r = int(np.searchsorted(cum, energy) + 1)
    else:
        cum = np.cumsum(S**2)/np.sum(S**2)
    V = VT.T  # [F, min(Ttr,F)]
    Phi = V[:, :r]  # spatial POD modes
    return mean.reshape(-1), Phi, S, cum

# ------------------------- Pseudo-time ordering (POD-based) ------------------------
def center_and_svd_rows(X_flat):
    mean = X_flat.mean(axis=0, keepdims=True); Xc = X_flat - mean
    U, S, VT = np.linalg.svd(Xc, full_matrices=False)
    return U, S, VT, mean

def build_sequence_by_pod(X, blend=1.0):
    N,H,W = X.shape
    X_flat = X.reshape(N, H*W)
    U,S,VT,mean = center_and_svd_rows(X_flat)
    if U.shape[1] == 1:
        scores = U[:,0]*S[0]
    else:
        s1 = U[:,0]*S[0]; s2 = U[:,1]*S[1]
        scores = float(blend)*s1 + (1.0-float(blend))*s2
    order = np.argsort(scores)
    return X[order], order

# ------------------------------- Exact DMD (discrete) ------------------------------
def fit_dmd(X_cols, energy=0.99, r=None):
    X  = X_cols[:, :-1]
    Xp = X_cols[:,  1:]
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    if r is None:
        e=S**2; cum=np.cumsum(e)/np.sum(e); r=int(np.searchsorted(cum, energy)+1)
    else:
        cum=np.cumsum(S**2)/np.sum(S**2)
    Ur=U[:, :r]; Sr=S[:r]; Vr=VT[:r,:].T; Sr_inv=np.diag(1.0/Sr)
    Atil = Ur.T @ Xp @ Vr @ Sr_inv
    eigvals, W = np.linalg.eig(Atil)
    Phi = (Xp @ Vr) @ Sr_inv @ W
    x0 = X_cols[:,0]; y0 = Ur.T @ x0; b0 = np.linalg.solve(W, y0)
    return {"Phi":Phi, "eigvals":eigvals, "W":W, "Ur":Ur, "r":r, "S":S, "cum":cum, "b0":b0}

def dmd_forecast_from(model, x_init_centered, steps):
    Phi, lam, Ur, W = model["Phi"], model["eigvals"], model["Ur"], model["W"]
    y_init = Ur.T @ x_init_centered
    b_init = np.linalg.solve(W, y_init)
    preds=[]
    for k in range(1, steps+1):
        coeff = (lam**k) * b_init
        preds.append(np.real(Phi @ coeff))
    return np.stack(preds, axis=1)  # [F, steps]

# ------------------------------ Utilities (plot, error) ----------------------------
def rmse(a, b):
    a=np.real(a); b=np.real(b)
    return float(np.sqrt(np.mean((a-b)**2)))

def shared_limits(a, b):
    both = np.concatenate([np.real(a).ravel(), np.real(b).ravel()])
    lo = np.percentile(both, 1); hi = np.percentile(both, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi<=lo:
        lo, hi = float(np.min(both)), float(np.max(both))
    return lo, hi

def imsave_pair(true_img, pred_img, path, title_left="True", title_right="Pred"):
    vmin, vmax = shared_limits(true_img, pred_img)
    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1); plt.imshow(np.real(true_img), origin="lower", vmin=vmin, vmax=vmax); plt.title(title_left); plt.colorbar()
    plt.subplot(1,2,2); plt.imshow(np.real(pred_img), origin="lower", vmin=vmin, vmax=vmax); plt.title(title_right); plt.colorbar()
    plt.tight_layout(); plt.savefig(path); plt.close()

# --------------------------------------- Main --------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="AirfRANS")
    ap.add_argument("--var", type=str, default="p", choices=["p","umag"])
    ap.add_argument("--M", type=int, default=60)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--W", type=int, default=128)
    ap.add_argument("--train_frac", type=float, default=0.8)
    # POD basis options
    ap.add_argument("--pod_energy", type=float, default=0.99)
    ap.add_argument("--pod_rank", type=int, default=None)
    # Sensor settings
    ap.add_argument("--sensors", type=int, default=50, help="number of point sensors")
    ap.add_argument("--sensor_seed", type=int, default=0)
    ap.add_argument("--noise_std", type=float, default=0.0, help="Gaussian noise std on sensors")
    ap.add_argument("--gamma", type=float, default=1e-6, help="Tikhonov reg for POD-only")
    ap.add_argument("--beta", type=float, default=1.0, help="nudging strength towards DMD prior")
    # DMD options
    ap.add_argument("--blend", type=float, default=0.8, help="POD ordering blend")
    ap.add_argument("--dmd_energy", type=float, default=0.99)
    ap.add_argument("--dmd_rank", type=int, default=None)
    ap.add_argument("--outdir", type=str, default="lab3_outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load rasters, build pseudo-time order (for DMD train/test)
    X = load_airfrans_as_rasters(args.root, args.var, args.M, args.H, args.W)  # [N,H,W]
    N,H,W = X.shape; F = H*W
    X_ord, order = build_sequence_by_pod(X, blend=args.blend)
    X_cols = X_ord.reshape(N, F).T  # [F,T]
    T = X_cols.shape[1]
    Ttr = max(10, int(np.floor(args.train_frac*T)))
    Tte = T - Ttr
    X_train = X_cols[:, :Ttr]   # [F,Ttr]
    X_test  = X_cols[:, Ttr:]   # [F,Tte]
    print(f"[split] T={T}, train={Ttr}, test={Tte}")

    # 2) POD basis from TRAIN (no time involved; use rows = snapshots)
    mu, Phi, S, cum = pod_from_train(X_train, r=args.pod_rank, energy=args.pod_energy)  # mu: [F], Phi: [F,r]
    r = Phi.shape[1]
    print(f"[pod] rank r = {r} (energy@r ≈ {cum[r-1]*100:.2f}%)")

    # 3) Choose a held-out test target (t+1 after train end)
    if Tte < 1:
        raise RuntimeError("No test horizon. Increase --M or reduce --train_frac.")
    j = 0  # first test step
    x_true  = X_test[:, j]                 # [F]
    x_true0 = x_true - mu                  # centered

    # 4) Create sensors (random pixel indices)
    rng = np.random.default_rng(args.sensor_seed)
    sensor_idx = rng.choice(F, size=min(args.sensors, F), replace=False)
    A = Phi[sensor_idx, :]                 # [M, r]
    y = x_true0[sensor_idx]
    if args.noise_std > 0:
        y = y + rng.normal(0.0, args.noise_std, size=y.shape)

    # 5) POD-only reconstruction from sensors: (A^T A + gamma I) a = A^T y
    ATA = A.T @ A
    rhs = A.T @ y
    a_pod = np.linalg.solve(ATA + args.gamma*np.eye(r), rhs)
    xhat0_pod = Phi @ a_pod
    xhat_pod  = xhat0_pod + mu
    rmse_pod = rmse(xhat_pod, x_true)
    print(f"[POD-only] sensors={len(sensor_idx)}, RMSE={rmse_pod:.4f}")

    # Save visualization
    imsave_pair(
        true_img=x_true.reshape(H,W),
        pred_img=xhat_pod.reshape(H,W),
        path=os.path.join(args.outdir, "recon_pod_only.png"),
        title_left="True (test t+1)",
        title_right=f"POD-from-sensors (K={len(sensor_idx)})"
    )

    # 6) DMD forecast from train end and project to POD for prior a0
    #    Mean removal in time for DMD:
    x_mean_time = X_train.mean(axis=1, keepdims=True)
    X_train0 = X_train - x_mean_time
    X_test0  = X_test  - x_mean_time
    model = fit_dmd(X_train0, energy=args.dmd_energy, r=args.dmd_rank)
    # Forecast to test step j (j+1 steps ahead from last train)
    preds0 = dmd_forecast_from(model, x_init_centered=X_train0[:, -1], steps=Tte)  # [F,Tte]
    xpred0 = preds0[:, j]                       # centered (DMD time-mean)
    # Bring to POD-centered coordinates: same mean as POD (mu). Align centers:
    # Convert DMD-centered pred to absolute, then re-center by POD mu.
    xpred_abs = xpred0 + x_mean_time[:,0]
    a0 = np.linalg.lstsq(Phi, xpred_abs - mu, rcond=None)[0]  # prior in POD basis

    # 7) Sensor nudging: (A^T A + (beta+gamma)I) a = A^T y + beta a0
    L = (args.beta + args.gamma)
    a_nudge = np.linalg.solve(ATA + L*np.eye(r), (A.T @ y) + args.beta * a0)
    xhat0_nudge = Phi @ a_nudge
    xhat_nudge  = xhat0_nudge + mu
    rmse_nudge = rmse(xhat_nudge, x_true)
    print(f"[Nudged]   beta={args.beta}, RMSE={rmse_nudge:.4f} (vs POD-only {rmse_pod:.4f})")

    imsave_pair(
        true_img=x_true.reshape(H,W),
        pred_img=xhat_nudge.reshape(H,W),
        path=os.path.join(args.outdir, "recon_nudged.png"),
        title_left="True (test t+1)",
        title_right=f"Nudged (K={len(sensor_idx)}, beta={args.beta})"
    )

    # 8) Optional: sweep K to see error vs sensors
    Ks = [5, 10, 20, 50, 100, 200] if len(range(1))==1 else []
    errs_pod=[]; errs_nudge=[]
    for K in Ks:
        if K>F: break
        idx = rng.choice(F, size=K, replace=False)
        A = Phi[idx,:]; y = x_true0[idx]
        a_pod = np.linalg.solve(A.T@A + args.gamma*np.eye(r), A.T@y)
        x_pod = (Phi@a_pod) + mu
        e_pod = rmse(x_pod, x_true)

        a_nu  = np.linalg.solve(A.T@A + (args.beta+args.gamma)*np.eye(r), A.T@y + args.beta*a0)
        x_nu  = (Phi@a_nu) + mu
        e_nu  = rmse(x_nu, x_true)
        errs_pod.append(e_pod); errs_nudge.append(e_nu)

    if Ks:
        plt.figure()
        plt.plot(Ks, errs_pod, marker='o', label='POD-only')
        plt.plot(Ks, errs_nudge, marker='o', label='Nudged')
        plt.xlabel("Number of sensors (K)"); plt.ylabel("RMSE"); plt.title("Error vs sensors")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "rmse_vs_sensors.png")); plt.close()

    # 9) Save where sensors were placed (for reference)
    np.save(os.path.join(args.outdir, "sensor_indices.npy"), sensor_idx)

    # 10) Eigenvalues plot (DMD)
    eigvals = model["eigvals"]
    plt.figure()
    th = np.linspace(0, 2*np.pi, 400)
    plt.plot(np.cos(th), np.sin(th), '--', linewidth=1)
    plt.scatter(eigvals.real, eigvals.imag, s=20)
    plt.axis('equal'); plt.xlabel("Re(λ)"); plt.ylabel("Im(λ)"); plt.title("DMD eigenvalues")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "dmd_eigs.png")); plt.close()

    print(f"[done] outputs in: {args.outdir}")

if __name__ == "__main__":
    main()
