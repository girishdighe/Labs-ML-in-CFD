# lab4_hankel_dmd_airfrans.py
# Time-Delay (Hankel) DMD with ridge stabilization on AirfRANS.
# - One file, no data generation.
# - Shared color scales for honest visuals.
# - Diagnostics: rep RMSE (Hankel), one-step (orig), long-horizon (orig), test mean RMSE vs copy-last.
# - Optional comparison to vanilla DMD from Lab 2.

import argparse, os
import numpy as np
import matplotlib.pyplot as plt

# ---------------- AirfRANS loading & rasterization ----------------
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
    data_list, _ = af.dataset.load(root=dataset_dir, task="full", train=True)
    M = min(M, len(data_list))
    print(f"[data] Using {M} snapshots from AirfRANS train split")

    xg = np.linspace(-2.0, 4.0, W)
    yg = np.linspace(-1.5, 1.5, H)
    Xg, Yg = np.meshgrid(xg, yg)

    rasters = []
    for i in range(M):
        sim = data_list[i]
        x, y = sim[:, 0], sim[:, 1]
        if var == "p":
            values = sim[:, 9]
        elif var == "umag":
            u, v = sim[:, 7], sim[:, 8]
            values = np.hypot(u, v)
        else:
            raise ValueError("var must be 'p' or 'umag'")
        Zi = griddata(np.c_[x, y], values, (Xg, Yg), method="linear")
        mask = np.isnan(Zi)
        if mask.any():
            Zi[mask] = griddata(np.c_[x, y], values, (Xg[mask], Yg[mask]), method="nearest")
        rasters.append(Zi.astype("float32"))
    return np.stack(rasters, axis=0)  # [N,H,W]

# ---------------- Pseudo-time ordering via POD scores ----------------
def center_and_svd_rows(X_flat):
    mean = X_flat.mean(axis=0, keepdims=True)
    Xc = X_flat - mean
    U, S, VT = np.linalg.svd(Xc, full_matrices=False)
    return U, S, VT, mean

def build_sequence_by_pod(X, blend=0.8):
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

# ---------------- Utilities (plot, error) ----------------
def rmse(a, b):
    a = np.real(a); b = np.real(b)
    return float(np.sqrt(np.mean((a - b)**2)))

def shared_limits(a, b):
    both = np.concatenate([np.real(a).ravel(), np.real(b).ravel()])
    lo = np.percentile(both, 1); hi = np.percentile(both, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(both)), float(np.max(both))
    return lo, hi

def imsave_pair(left_img, right_img, path, title_left="Left", title_right="Right"):
    vmin, vmax = shared_limits(left_img, right_img)
    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1); plt.imshow(np.real(left_img), origin="lower", vmin=vmin, vmax=vmax); plt.title(title_left); plt.colorbar()
    plt.subplot(1,2,2); plt.imshow(np.real(right_img), origin="lower", vmin=vmin, vmax=vmax); plt.title(title_right); plt.colorbar()
    plt.tight_layout(); plt.savefig(path); plt.close()

# ---------------- Vanilla DMD (optional comparison) ----------------
def fit_dmd_vanilla(X_cols, energy=0.99, r=None, ridge=0.0):
    X  = X_cols[:, :-1]; Xp = X_cols[:, 1:]
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    if r is None:
        e=S**2; cum=np.cumsum(e)/np.sum(e); r=int(np.searchsorted(cum, energy)+1)
    Ur = U[:,:r]; Sr = S[:r]; Vr = VT[:r,:].T
    if ridge > 0:
        Y = Ur.T @ Xp; Xr = Ur.T @ X
        Atil = Y @ Xr.T @ np.linalg.inv(Xr @ Xr.T + ridge*np.eye(r))
    else:
        Sr_inv = np.diag(1.0/Sr)
        Atil = Ur.T @ Xp @ Vr @ Sr_inv
    eigvals, Weig_v = np.linalg.eig(Atil)
    Sr_inv = np.diag(1.0/Sr)
    Phi = (Xp @ Vr) @ Sr_inv @ Weig_v
    x0 = X_cols[:,0]; y0 = Ur.T @ x0; b0 = np.linalg.solve(Weig_v, y0)
    return {"Phi":Phi, "eigvals":eigvals, "Ur":Ur, "Weig":Weig_v, "r":r, "b0":b0}

def forecast_dmd_vanilla(model, x_init_centered, steps):
    Phi, lam, Ur, Weig_v = model["Phi"], model["eigvals"], model["Ur"], model["Weig"]
    y_init = Ur.T @ x_init_centered
    b_init = np.linalg.solve(Weig_v, y_init)
    preds = []
    for k in range(1, steps+1):
        coeff = (lam**k) * b_init
        preds.append(np.real(Phi @ coeff))
    return np.stack(preds, axis=1)

# ---------------- Hankel (time-delay) DMD ----------------
def build_hankel_pairs(X_cols0, q=6, stride=1):
    """
    X_cols0: [F,T] centered sequence. Return Z0,Z1 and z_init (last train window).
    """
    F, T = X_cols0.shape
    L = T - (q-1)*stride
    if L < 2:
        raise ValueError("Not enough snapshots for the chosen q/stride.")
    K = L - 1
    qF = q*F
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

def fit_hankel_dmd(X_cols, q=6, stride=1, energy=0.99, r=None, ridge=0.0):
    """
    Fit DMD on block-Hankel pairs built from CENTERED X_cols.
    Ridge applies in reduced coordinates; modes use Z1,Vr,Sr_inv consistently.
    """
    Z0, Z1, z_init = build_hankel_pairs(X_cols, q=q, stride=stride)
    U, S, VT = np.linalg.svd(Z0, full_matrices=False)
    if r is None:
        e=S**2; cum=np.cumsum(e)/np.sum(e); r=int(np.searchsorted(cum, energy)+1)
    Ur = U[:,:r]; Sr = S[:r]; Vr = VT[:r,:].T
    Sr_inv = np.diag(1.0/Sr)

    if ridge > 0:
        # Reduced ridge regression
        Xr = Ur.T @ Z0
        Yr = Ur.T @ Z1
        Atil = Yr @ Xr.T @ np.linalg.inv(Xr @ Xr.T + ridge*np.eye(r))
    else:
        Atil = Ur.T @ Z1 @ Vr @ Sr_inv

    eigvals, Weig = np.linalg.eig(Atil)
    # Exact DMD modes in Hankel space (consistent for both branches)
    Phi = (Z1 @ Vr) @ Sr_inv @ Weig

    # Stable init
    y0 = Ur.T @ z_init
    b0 = np.linalg.solve(Weig, y0)

    model = {
        "Phi": Phi, "eigvals": eigvals, "Ur": Ur, "Weig": Weig,
        "r": r, "q": q, "F": X_cols.shape[0], "z_init": z_init, "ridge": ridge
    }
    return model

def forecast_hankel_dmd(model, steps):
    Phi, lam, Ur, Weig = model["Phi"], model["eigvals"], model["Ur"], model["Weig"]
    z_init = model["z_init"]; F = model["F"]; q = model["q"]
    y_init = Ur.T @ z_init
    b_init = np.linalg.solve(Weig, y_init)
    preds_lastblock = []
    for k in range(1, steps+1):
        coeff = (lam**k) * b_init
        z_k = np.real(Phi @ coeff)
        last = z_k[(q-1)*F : q*F]  # last block is the new snapshot
        preds_lastblock.append(last)
    return np.stack(preds_lastblock, axis=1)  # [F, steps]

# ------------------------------ Main ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="AirfRANS")
    ap.add_argument("--var", type=str, default="p", choices=["p","umag"])
    ap.add_argument("--M", type=int, default=60)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--W", type=int, default=128)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--blend", type=float, default=0.8)
    # Hankel params
    ap.add_argument("--q", type=int, default=6)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--energy", type=float, default=0.99)
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--ridge", type=float, default=1e-3)
    # Compare vanilla DMD?
    ap.add_argument("--compare_vanilla", action="store_true")
    ap.add_argument("--vanilla_rank", type=int, default=None)
    ap.add_argument("--vanilla_ridge", type=float, default=0.0)
    ap.add_argument("--outdir", type=str, default="lab4_outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load & rasterize
    X = load_airfrans_as_rasters(args.root, args.var, args.M, args.H, args.W)
    N, Hpx, Wpx = X.shape
    F = Hpx * Wpx

    # 2) Pseudo-time ordering and column form
    X_ord, _ = build_sequence_by_pod(X, blend=args.blend)
    X_cols = X_ord.reshape(N, F).T     # [F, T]
    T = X_cols.shape[1]

    # 3) Split + mean removal
    Ttr = max(10, int(np.floor(args.train_frac * T))); Tte = T - Ttr
    X_train = X_cols[:, :Ttr]; X_test = X_cols[:, Ttr:]
    x_mean = X_train.mean(axis=1, keepdims=True)
    X_train0 = X_train - x_mean
    X_test0  = X_test  - x_mean
    print(f"[split] T={T}, train={Ttr}, test={Tte}")

    # 4) Fit Hankel-DMD
    model = fit_hankel_dmd(X_train0, q=args.q, stride=args.stride,
                           energy=args.energy, r=args.rank, ridge=args.ridge)
    r = model["r"]
    print(f"[hankel-dmd] q={args.q}, rank r={r}, ridge={args.ridge:g}")

    # Diagnostics: representational capacity in Hankel space
    Z0, Z1, _ = build_hankel_pairs(X_train0, q=args.q, stride=args.stride)
    Ur = model["Ur"]
    Z0_hat = Ur @ (Ur.T @ Z0)
    rep_rmse_hankel = rmse(Z0_hat, Z0)
    print(f"[train] Hankel representational RMSE: {rep_rmse_hankel:.4f}")

    # One-step in original space: predict first test frame from last train window
    preds1 = forecast_hankel_dmd(model, steps=1)[:, 0]  # [F]
    one_step_rmse = rmse(preds1, X_test0[:, 0])
    print(f"[train->test] One-step RMSE (orig): {one_step_rmse:.4f}")

    # 5) Test mean RMSE vs copy-last baseline (+ visuals)
    if Tte > 0:
        preds_all = forecast_hankel_dmd(model, steps=Tte)  # [F, Tte]
        rmse_list = []
        copy_list = []
        x_last_train0 = X_train0[:, -1]
        for j in range(Tte):
            gt0 = X_test0[:, j]
            pr0 = preds_all[:, j]
            rmse_list.append(rmse(pr0, gt0))
            copy_list.append(rmse(x_last_train0, gt0))
            if j in [0, min(1, Tte-1)]:
                imsave_pair(
                    (gt0 + x_mean[:,0]).reshape(Hpx, Wpx),
                    (pr0 + x_mean[:,0]).reshape(Hpx, Wpx),
                    os.path.join(args.outdir, f"hankel_forecast_pair_{j}.png"),
                    title_left=f"True[t+{j+1}]",
                    title_right=f"Hankel-DMD Pred[t+{j+1}]"
                )
        print(f"[test] mean RMSE over {Tte} steps: Hankel-DMD={np.mean(rmse_list):.6f} | copy-last={np.mean(copy_list):.6f}")
    else:
        print("[test] No test horizon.")

    # 6) Train-end one-step visual (diagnostic)
    if Tte > 0:
        true_train_end = (X_train0[:, -1] + x_mean[:,0]).reshape(Hpx, Wpx)
        imsave_pair(
            true_train_end,
            (preds1 + x_mean[:,0]).reshape(Hpx, Wpx),
            os.path.join(args.outdir, "hankel_one_step_train_end.png"),
            title_left="True[train end (t+1)]",
            title_right="Hankel-DMD one-step from last window"
        )

    # 7) Eigenvalues
    eig = model["eigvals"]
    plt.figure()
    th = np.linspace(0, 2*np.pi, 400)
    plt.plot(np.cos(th), np.sin(th), '--', linewidth=1)
    plt.scatter(eig.real, eig.imag, s=20)
    plt.axis('equal'); plt.xlabel("Re(λ)"); plt.ylabel("Im(λ)")
    plt.title("Hankel-DMD eigenvalues (discrete)")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "hankel_dmd_eigs.png")); plt.close()

    # 8) Optional: compare to vanilla DMD
    if args.compare_vanilla and Tte>0:
        vmodel = fit_dmd_vanilla(X_train0, energy=args.energy, r=args.vanilla_rank, ridge=args.vanilla_ridge)
        vpreds = forecast_dmd_vanilla(vmodel, x_init_centered=X_train0[:, -1], steps=Tte)
        v_rmse = np.mean([rmse(vpreds[:,j], X_test0[:,j]) for j in range(Tte)])
        print(f"[compare] Vanilla DMD mean RMSE: {v_rmse:.6f}")

    print(f"[done] outputs in: {args.outdir}")

if __name__ == "__main__":
    main()