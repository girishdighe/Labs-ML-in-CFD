# lab2_dmd_airfrans.py
# Dynamic Mode Decomposition (DMD) on AirfRANS using a pseudo-time sequence.
# - One file, no data generation.
# - Mean removal on TRAIN split.
# - Numerically stable DMD amplitudes.
# - Real-safe math/plots.
# - Shared color scales for apples-to-apples visuals.
# - Extra diagnostics: representational RMSE and one-step RMSE.

import argparse, os
import numpy as np
import matplotlib.pyplot as plt

# ------------------------- AirfRANS loading & rasterization -------------------------
def ensure_airfrans(root):
    """
    Ensure the preprocessed AirfRANS dataset is present (no OpenFOAM files).
    Downloads once if missing.
    """
    import airfrans as af
    dataset_dir = os.path.join(root, "Dataset")
    if not os.path.exists(dataset_dir):
        print("[setup] Downloading AirfRANS (preprocessed, no OpenFOAM files)…")
        af.dataset.download(root=root, file_name="Dataset", unzip=True, OpenFOAM=False)
    return dataset_dir

def load_airfrans_as_rasters(root, var="p", M=60, H=128, W=128):
    """
    Load M AirfRANS snapshots and rasterize one variable to a fixed grid.
      var='p'    -> p_over_rho (index 9)
      var='umag' -> sqrt(u^2+v^2) from columns 7,8
    Returns X with shape [N, H, W], float32.
    """
    import airfrans as af
    from scipy.interpolate import griddata

    dataset_dir = ensure_airfrans(root)
    data_list, names = af.dataset.load(root=dataset_dir, task="full", train=True)
    M = min(M, len(data_list))
    print(f"[data] Using {M} snapshots from AirfRANS train split")

    # Fixed Eulerian grid covering AirfRANS cropped domain
    xg = np.linspace(-2.0, 4.0, W)
    yg = np.linspace(-1.5, 1.5, H)
    Xg, Yg = np.meshgrid(xg, yg)

    rasters = []
    for i in range(M):
        sim = data_list[i]  # (N_pts, 12)
        x, y = sim[:, 0], sim[:, 1]
        if var == "p":
            values = sim[:, 9]  # p_over_rho
        elif var == "umag":
            u, v = sim[:, 7], sim[:, 8]
            values = np.hypot(u, v)
        else:
            raise ValueError("var must be 'p' or 'umag'")

        Zi = griddata(points=np.c_[x, y], values=values, xi=(Xg, Yg), method="linear")
        # Fill any NaNs by nearest to avoid holes
        mask = np.isnan(Zi)
        if mask.any():
            Zi[mask] = griddata(np.c_[x, y], values, (Xg[mask], Yg[mask]), method="nearest")
        rasters.append(Zi.astype("float32"))

    return np.stack(rasters, axis=0)  # [N,H,W]

# ------------------------- Pseudo-time ordering via POD (PCA) -----------------------
def center_and_svd_rows(X_flat):
    """
    X_flat: [N, F] (rows are snapshots). Returns U,S,VT,mean (centering by feature means).
    """
    mean = X_flat.mean(axis=0, keepdims=True)  # [1, F]
    Xc = X_flat - mean
    U, S, VT = np.linalg.svd(Xc, full_matrices=False)  # Xc = U S VT
    return U, S, VT, mean

def build_sequence_by_pod(X, blend=1.0):
    """
    Build a pseudo-time order by sorting snapshots along a POD path.
      blend=1.0 uses only 1st POD score; 0.8 means 0.8*POD1 + 0.2*POD2 (smoother path).
    Returns X_sorted [N,H,W] and the order indices.
    """
    N, H, W = X.shape
    X_flat = X.reshape(N, H*W)           # [N, F]
    U, S, VT, mean = center_and_svd_rows(X_flat)
    if U.shape[1] == 1:
        scores = U[:, 0] * S[0]
    else:
        w1 = float(blend)
        w2 = 1.0 - w1
        s1 = U[:, 0] * S[0]
        s2 = U[:, 1] * S[1]
        scores = w1 * s1 + w2 * s2
    order = np.argsort(scores)           # ascending path
    return X[order], order

# ------------------------------- Exact DMD (discrete) ------------------------------
def choose_rank_by_energy(S, energy=0.99):
    e = S**2
    cum = np.cumsum(e) / np.sum(e)
    r = int(np.searchsorted(cum, energy) + 1)
    return r, cum

def fit_dmd(X_cols, energy=0.99, r=None):
    """
    Fit Exact DMD on a *mean-removed* sequence X_cols with shape [F, T].
    Returns dictionary with DMD factors and diagnostics.
    """
    # One-step shift pair
    X  = X_cols[:, :-1]   # [F, T-1]
    Xp = X_cols[:,  1:]   # [F, T-1]

    # Economy SVD of X
    U, S, VT = np.linalg.svd(X, full_matrices=False)  # X = U S VT
    if r is None:
        r, cum = choose_rank_by_energy(S, energy)
    else:
        cum = np.cumsum(S**2)/np.sum(S**2)

    Ur = U[:, :r]              # [F, r]
    Sr = S[:r]                 # [r]
    Vr = VT[:r, :].T           # [T-1, r]
    Sr_inv = np.diag(1.0 / Sr)

    # Reduced operator: A_tilde = Ur^T X' Vr Σr^{-1}
    A_tilde = Ur.T @ Xp @ Vr @ Sr_inv    # [r, r]

    # Eigen-decomposition in reduced space
    eigvals, W = np.linalg.eig(A_tilde)  # A_tilde W = W Λ

    # Exact DMD modes in full space
    Phi = (Xp @ Vr) @ Sr_inv @ W         # [F, r]

    # Stable initial amplitudes via reduced coordinates (not Phi^+ x0)
    x0  = X_cols[:, 0]                   # first (centered) snapshot
    y0  = Ur.T @ x0                      # reduced coords
    b0  = np.linalg.solve(W, y0)         # W b0 = y0

    return {
        "Phi": Phi,           # [F, r]
        "eigvals": eigvals,   # [r]
        "W": W,               # [r, r]
        "Ur": Ur,             # [F, r]
        "r": r,
        "S": S,
        "cum_energy": cum,
        "b0": b0,
    }

def dmd_reconstruct(model, k):
    """
    Reconstruct snapshot k (k=0 is x0) from the DMD model in *centered* space.
    """
    Phi, lam, b0 = model["Phi"], model["eigvals"], model["b0"]
    return np.real(Phi @ ((lam**k) * b0))

def dmd_forecast_from(model, x_init_centered, steps):
    """
    Forecast 'steps' steps ahead starting from an arbitrary *centered* state.
    Returns array [F, steps] in centered space.
    """
    Phi, lam, Ur, W = model["Phi"], model["eigvals"], model["Ur"], model["W"]
    y_init = Ur.T @ x_init_centered
    b_init = np.linalg.solve(W, y_init)
    preds = []
    for k in range(1, steps+1):
        coeff = (lam**k) * b_init
        preds.append(np.real(Phi @ coeff))
    return np.stack(preds, axis=1)  # [F, steps]

# ------------------------------ Utilities (plot, error) ----------------------------
def rmse(a, b):
    a = np.real(a); b = np.real(b)
    return float(np.sqrt(np.mean((a - b)**2)))

def _shared_vmin_vmax(a, b):
    a = np.real(a); b = np.real(b)
    both = np.concatenate([a.ravel(), b.ravel()])
    lo = np.percentile(both, 1)
    hi = np.percentile(both, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi <= lo):
        lo, hi = float(np.min(both)), float(np.max(both))
    return lo, hi

def imsave_pair(true_img, pred_img, path, title_left="True", title_right="Pred"):
    a = np.real(true_img); b = np.real(pred_img)
    vmin, vmax = _shared_vmin_vmax(a, b)
    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1); plt.imshow(a, origin="lower", vmin=vmin, vmax=vmax); plt.title(title_left); plt.colorbar()
    plt.subplot(1,2,2); plt.imshow(b, origin="lower", vmin=vmin, vmax=vmax); plt.title(title_right); plt.colorbar()
    plt.tight_layout(); plt.savefig(path); plt.close()

# ------------------------------------------ Main -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="AirfRANS")
    ap.add_argument("--var", type=str, default="p", choices=["p","umag"])
    ap.add_argument("--M", type=int, default=60)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--W", type=int, default=128)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--energy", type=float, default=0.99)
    ap.add_argument("--rank", type=int, default=None, help="Override DMD rank; else use --energy")
    ap.add_argument("--blend", type=float, default=1.0, help="POD ordering blend: 1.0=POD1 only, 0.8=80% POD1 + 20% POD2")
    ap.add_argument("--outdir", type=str, default="dmd_outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load & rasterize AirfRANS → [N,H,W]
    X = load_airfrans_as_rasters(
        root=args.root, var=args.var, M=args.M, H=args.H, W=args.W
    )
    N, H, W = X.shape
    F = H * W

    # 2) Pseudo-time ordering by blended POD scores
    X_ord, order = build_sequence_by_pod(X, blend=args.blend)  # [N,H,W]
    X_cols = X_ord.reshape(N, F).T                              # [F, T], T=N
    T = X_cols.shape[1]

    # 3) Train/test split along this ordering
    T_train = max(10, int(np.floor(args.train_frac * T)))
    T_test  = T - T_train
    X_train = X_cols[:, :T_train]   # [F, T_train]
    X_test  = X_cols[:, T_train:]   # [F, T_test]
    print(f"[split] T={T}, train={T_train}, test={T_test}")

    # 4) Mean removal on the TRAIN split only (temporal mean)
    x_mean = X_train.mean(axis=1, keepdims=True)  # [F,1]
    X_train0 = X_train - x_mean
    X_test0  = X_test  - x_mean

    # 5) Fit DMD on centered train sequence
    model = fit_dmd(X_train0, energy=args.energy, r=args.rank)
    r = model["r"]
    print(f"[dmd] rank r = {r}")

    # 6) Diagnostics: representational capacity (no dynamics)
    Phi = model["Phi"]                                 # [F, r]
    B = np.linalg.lstsq(Phi, X_train0, rcond=None)[0]  # [r, T_train]
    Xhat_train0 = Phi @ B                              # [F, T_train]
    rep_rmse = rmse(Xhat_train0, X_train0)
    print(f"[train] Representational RMSE (Phi * Phi^+ X): {rep_rmse:.4f}")

    # 7) Diagnostics: one-step linearity error on train
    X  = X_train0[:, :-1]
    Xp = X_train0[:,  1:]
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    r_use = min(r, len(S))
    Ur = U[:, :r_use]; Sr = S[:r_use]; Vr = VT[:r_use, :].T
    Sr_inv = np.diag(1.0 / Sr)
    A_tilde = Ur.T @ Xp @ Vr @ Sr_inv
    A = Ur @ A_tilde @ Ur.T
    one_step_rmse = rmse(A @ X, Xp)
    print(f"[train] One-step RMSE (||X' - A X||): {one_step_rmse:.4f}")

    # 8) Sanity: reconstruct last train snapshot (centered space)
    x_last_true0 = X_train0[:, -1]
    x_last_pred0 = dmd_reconstruct(model, k=T_train-1)
    train_rmse_last = rmse(x_last_pred0, x_last_true0)
    print(f"[train] RMSE at last train snapshot ≈ {train_rmse_last:.6f}")

    # Visual: add mean back for plotting
    imsave_pair(
        true_img=(x_last_true0 + x_mean[:,0]).reshape(H, W),
        pred_img=(x_last_pred0 + x_mean[:,0]).reshape(H, W),
        path=os.path.join(args.outdir, "train_last_pair.png"),
        title_left="True[train end]",
        title_right="DMD Recon[train end]"
    )

    # 9) Forecast the test horizon (if any), compare to copy-last baseline
    if T_test > 0:
        preds0 = dmd_forecast_from(model, x_init_centered=x_last_true0, steps=T_test)  # [F, T_test]
        test_rmse_list = []
        copylast_rmse_list = []
        for j in range(T_test):
            gt0 = X_test0[:, j]
            pr0 = preds0[:, j]
            test_rmse_list.append(rmse(pr0, gt0))
            copylast_rmse_list.append(rmse(x_last_true0, gt0))

            # Save a couple of visual comparisons with mean added back
            if j in [0, min(1, T_test-1)]:
                imsave_pair(
                    true_img=(gt0 + x_mean[:,0]).reshape(H, W),
                    pred_img=(pr0 + x_mean[:,0]).reshape(H, W),
                    path=os.path.join(args.outdir, f"forecast_pair_{j}.png"),
                    title_left=f"True[t+{j+1}]",
                    title_right=f"DMD Pred[t+{j+1}]"
                )

        print(f"[test] mean RMSE over {T_test} steps: "
              f"DMD={np.mean(test_rmse_list):.6f} | copy-last={np.mean(copylast_rmse_list):.6f}")
    else:
        print("[test] No test horizon—reduce train_frac or increase M.")

    # 10) Eigenvalues (discrete-time) — stability / oscillations
    eigvals = model["eigvals"]
    plt.figure()
    theta = np.linspace(0, 2*np.pi, 400)
    plt.plot(np.cos(theta), np.sin(theta), '--', linewidth=1)  # unit circle
    plt.scatter(eigvals.real, eigvals.imag, s=20)
    plt.xlabel("Re(λ)"); plt.ylabel("Im(λ)"); plt.axis('equal'); plt.title("DMD eigenvalues (discrete)")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "dmd_eigs.png")); plt.close()

    print(f"[done] outputs written to: {args.outdir}")

    # --- EXTRA VIS 1: Pure projection of last train frame (no dynamics) ---
    # Shows representational capacity: X̂_last = Φ (Φ^+ x_last)
    x_last_true0 = X_train0[:, -1]
    b_last = np.linalg.lstsq(Phi, x_last_true0, rcond=None)[0]
    x_last_proj0 = Phi @ b_last
    imsave_pair(
        true_img=(x_last_true0 + x_mean[:,0]).reshape(H, W),
        pred_img=(x_last_proj0 + x_mean[:,0]).reshape(H, W),
        path=os.path.join(args.outdir, "train_last_projection_pair.png"),
        title_left="True[train end]",
        title_right="Projection via Φ (no dynamics)"
    )

    # --- EXTRA VIS 2: One-step prediction at train end ---
    # Build full-space A and do one-step: x_pred = A x_{T-1}
    X  = X_train0[:, :-1]; Xp = X_train0[:, 1:]
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    r_use = min(r, len(S))
    Ur = U[:, :r_use]; Sr = S[:r_use]; Vr = VT[:r_use, :].T
    Sr_inv = np.diag(1.0 / Sr)
    A_tilde = Ur.T @ Xp @ Vr @ Sr_inv
    A = Ur @ A_tilde @ Ur.T

    x_prev0      = X_train0[:, -2]
    x_last_1step = A @ x_prev0
    imsave_pair(
        true_img=(x_last_true0 + x_mean[:,0]).reshape(H, W),
        pred_img=(x_last_1step + x_mean[:,0]).reshape(H, W),
        path=os.path.join(args.outdir, "train_last_onestep_pair.png"),
        title_left="True[train end]",
        title_right="One-step from previous (A x_{t-1})"
    )

if __name__ == "__main__":
    main()
