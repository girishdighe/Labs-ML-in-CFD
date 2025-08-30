# Lab 4 — Time-Delay (Hankel) DMD with Stabilized Forecasting on AirfRANS

# Abstract
We address long-horizon drift that appears when vanilla Dynamic Mode Decomposition (DMD) is fitted on AirfRANS snapshots sorted along a POD-based pseudo-time. We lift the state to a time-delay (Hankel) embedding and learn a single linear map in the delay space (Koopman/DMD view of an AR(q) model). A small ridge in the reduced least-squares fit shrinks eigenvalues toward the unit circle and improves rollout stability. On 128×128 pressure rasters with M=60 ordered snapshots, train/test split 48/12, delay length q=6, rank r=12, ridge 10−3, Hankel-DMD achieves mean test RMSE = 127.24, beating copy-last = 185.23 and vanilla DMD = 180.12. One-step RMSE equals 127.24, indicating uniform error over the horizon. Eigenvalues cluster inside/near the unit circle with a small group near +1, consistent with slow, neutral drift and stable forecasts. Qualitative frames at t+1 and t+2 match suction and recovery regions closely.

# 1. Introduction
Vanilla DMD fits a linear map xk+1 ⁣≈ ⁣Axk on a sequence of high-dimensional states xk∈RF. When the sequence is a pseudo-time path (as with AirfRANS shape/condition sweeps), local nonlinear curvature is folded into a single step and small one-step biases accumulate over many steps. Time-delay DMD mitigates this by giving the linear model memory: we evolve a stacked vector of the last q states. This is equivalent to a Koopman-linear AR model and typically reduces drift without abandoning interpretability.

# 2. Data and Preprocessing
Dataset & variable. AirfRANS training cases (no OpenFOAM runs). Field: p/ρ.
Rasterization. Interpolate the unstructured points (x,y)→p/ρ onto a Cartesian grid [−2,4]×[−1.5,1.5] with H×W=128×128, then vectorize to x∈RF, F=16384.
Pseudo-time ordering. Compute SVD of the row-centered snapshot matrix and sort snapshots by a blended POD score s=αs1+(1−α)s2 with α=0.8.
Split & centering. Order produces T=60 snapshots, split into train Ttr=48 and test Tte=12. Remove the train temporal mean:
Xtrain0=Xtrain−xˉ1⊤,Xtest0=Xtest−xˉ1⊤.

# 3. Method
## 3.1 Time-delay (Hankel) embedding
Form delay vectors by stacking q consecutive centered states:
zk=[xkxk+1⋮xk+q−1]∈RqF.
Build block-Hankel training pairs
Z0=[z1,…,zK],Z1=[z2,…,zK+1],K=Ttr−(q−1)−1.
Learning a single linear map zk+1 ⁣≈ ⁣A~zk is the Koopman/DMD form of an AR(q) model in the original space.
## 3.2 Exact DMD in delay space
Compute reduced SVD Z0=UΣV⊤ and truncate to rank r (by fixed value or energy). The reduced operator is
A~=Ur⊤Z1VrΣr−1∈Rr×r,
with eigendecomposition A~W=WΛ. The exact Hankel-DMD modes and amplitudes are
Φ=Z1VrΣr−1W,b0=W−1(Ur⊤zinit),
where zinit is the last train window. A k-step forecast in delay space is zk≈ΦΛkb0. The new physical snapshot is the last block of zk.
## 3.3 Stabilization (ridge in reduced coordinates)
To temper drift we solve a ridge-regularized reduced LS in the Ur coordinates:
A~=YX⊤ (XX⊤+αI)−1,X=Ur⊤Z0,  Y=Ur⊤Z1,
with a small α>0. This shrinks eigenvalues slightly toward the unit circle without destroying interpretability.

# 4. Experimental Settings
Grid: 128×128, variable p/ρ.
Snapshots: M=60 after ordering; split 48/12.
Delay length: q=6 (stride 1).
Rank: r=12 (fixed).
Ridge: α=10−3.
Comparison: vanilla DMD (rank 12) on the same centered, ordered sequence.
Command.
python lab4_hankel_dmd_airfrans.py --root AirfRANS --var p --M 60 --H 128 --W 128 \
  --train_frac 0.8 --blend 0.8 --q 6 --rank 12 --ridge 1e-3 --energy 0.99 \
  --compare_vanilla --vanilla_rank 12


# 5. Metrics
Hankel representational RMSE: ∥Z0−UrUr⊤Z0∥RMS → capacity of the delay subspace.
One-step RMSE (orig): error of the first predicted physical snapshot xt+1.
Mean test RMSE: average RMSE over all Tte test steps, compared to copy-last and vanilla DMD.
Qualitative frames: paired True vs Pred at t+1, t+2; one-step at the train end.
Spectrum: discrete eigenvalues on the complex plane with unit circle overlay.

# 6. Results
Scalars.
Hankel representational RMSE: 17.2909
One-step RMSE (orig): 127.2381
Mean test RMSE (12 steps): Hankel-DMD = 127.2351, copy-last = 185.2315, vanilla DMD = 180.1185
Gains.
vs copy-last: 31.3% lower error
vs vanilla DMD: 29.4% lower error
Spectral picture. Eigenvalues lie inside/near the unit circle with a small cluster near +1 on the real axis—neutral/slow modes with mild oscillation. This is the desired “stable but expressive” regime for multi-step rollouts.
Visuals.
t+1 and t+2 panels: suction bubble magnitude/location and recovery ridge downstream are reproduced closely; off-body gradients align well.
One-step at train end (from the last window): consistent with the scalar one-step RMSE; no obvious phase flip or sign inversions.
(Your figures: hankel_forecast_pair_0.png, hankel_forecast_pair_1.png, hankel_one_step_train_end.png, hankel_dmd_eigs.png.)

# 7. Interpretation
## 7.1 Why delay embedding helps here
The AirfRANS “time” axis is a manifold path through design/condition space, not physical time. A single-step map has to linearize curved motion on this path, which creates small systematic biases that accumulate. The delay vector zk=[xk;…;xk+q−1] encodes local curvature; one linear step in z-space induces a higher-order update in x-space (AR(q)). This reduces phase/amplitude drift across the test horizon.
## 7.2 Why the ridge matters
The ridge solution shrinks A~’s eigenvalues modestly, counteracting the tendency of near-unit eigenvalues to drift. Your spectrum shows precisely that: eigenvalues pulled slightly inward, but not overdamped.
## 7.3 Uniform error across the horizon
One-step RMSE ≈ mean test RMSE. That indicates no catastrophic growth with horizon length—i.e., the model’s local bias is roughly constant and stable under iteration.

# 8. Sensitivity (what is most worth trying next)
You can often squeeze a few more points by a tiny grid search:
Delay length q: 6–8 is a sweet spot; try --q 8.
Ridge α: 5e-4 or 2e-3 (too large overdamps, too small can drift).
Rank r: try 14 or 16; delay embeddings benefit from a touch more capacity.
More snapshots: --M 80 improves conditioning of the Hankel pairs.
Example:
python lab4_hankel_dmd_airfrans.py --root AirfRANS --var p --M 80 --H 128 --W 128 \
  --train_frac 0.8 --blend 0.8 --q 8 --rank 14 --ridge 5e-4 \
  --compare_vanilla --vanilla_rank 12

Keep the spectrum close to (or just inside) the unit circle and judge by mean test RMSE first, not only one-step.

# 9. Limitations and extensions
Pseudo-time ≠ physics. Delay DMD stabilizes rollouts but does not recover causal dynamics. If you later get actual time-resolved data, repeat the procedure; improvements will be larger.
Fixed grid & single variable. Extending to (u,v,p) requires channel standardization and either concatenation or block-structured modes.
Linear prior. Further drift reductions are possible with time-delay DMDc (exogenous inputs) or a tiny residual network on top of the linear core, while keeping interpretability.

# 10. Conclusions
Time-delay (Hankel) DMD with a small reduced-space ridge provides a simple, interpretable, and effective fix for long-horizon drift on AirfRANS pseudo-time sequences. Relative to vanilla DMD and a naïve baseline, it achieves ~30% lower mean test error, yields stable spectra, and produces visually faithful short-horizon forecasts. This is the right “next rung” beyond POD/DMD before deploying heavier neural models.

# 11. Reproducibility & Artifacts
Command used
python lab4_hankel_dmd_airfrans.py --root AirfRANS --var p --M 60 --H 128 --W 128 \
  --train_frac 0.8 --blend 0.8 --q 6 --rank 12 --ridge 1e-3 --energy 0.99 \
  --compare_vanilla --vanilla_rank 12

Generated assets
hankel_forecast_pair_0.png — True vs Pred at t+1
hankel_forecast_pair_1.png — True vs Pred at t+2
hankel_one_step_train_end.png — one-step from last train window
hankel_dmd_eigs.png — eigenvalues on unit circle
Console metrics printed exactly as reported above

# Appendix A — Key derivations (compact)
Exact modes in delay space.
From Z0=UΣV⊤, truncate to r, define A~=Ur⊤Z1VrΣr−1. With A~W=WΛ,
Z1≈UrA~Ur⊤Z0=UrWΛW−1Ur⊤Z0.
Right-multiplying by VrΣr−1 and noting Z0VrΣr−1≈Ur gives
Φ=Z1VrΣr−1W,
which are the exact DMD modes in the Hankel state space.
Ridge in reduced coordinates.
Minimizing ∥Y−A~X∥F2+α∥A~∥F2 yields
A~=YX⊤(XX⊤+αI)−1.
This is equivalent to Tikhonov on the operator, gently shrinking eigenvalues of A~.



