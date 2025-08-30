# Lab 2 — Dynamic Mode Decomposition (DMD) on AirfRANS (pseudo-time)

# Abstract
We estimate a linear evolution model for airfoil pressure fields using Dynamic Mode Decomposition (DMD). Because AirfRANS provides steady snapshots (no physical time), we construct a pseudo-time ordering by sorting snapshots along a blended POD coordinate and then fit Exact DMD on the mean-removed training sequence. We report four complementary metrics: (i) representational RMSE (how well DMD modes span the data), (ii) one-step RMSE(linearity of the sequence), (iii) long-horizon train-end RMSE, and (iv) test mean RMSE compared to a copy-last baseline. With rank=12, blend=0.8, M=60, we obtain: representational RMSE 6.00, one-step RMSE 65.45, long-horizon train-end RMSE 378.26, and test mean RMSE 180.12 vs 185.23 (copy-last). Eigenvalues cluster inside/near the unit circle with one near unity, consistent with slow, nearly neutral drift along the snapshot manifold. Forecasts beat the baseline and short-horizon visuals are faithful; long-horizon differences are explained by error accumulation in a non-physical sequence.

# 1. Problem Setup
Data and variable
Dataset: AirfRANS (train/full split).
Field: p/ρ (static pressure divided by density), interpolated to a fixed Eulerian grid (H=W=128) over [x,y]∈[−2,4]×[−1.5,1.5].
Snapshots: N=60 rasters (we use 80% for “train time” and 20% for “test time”).
Why rasterize? DMD requires a consistent state vector. We interpolate unstructured node values to a common grid (linear interpolation with nearest fallback to avoid holes).
Pseudo-time ordering by POD
Let the snapshot matrix (rows are snapshots, columns are pixels) be X∈RN×F, F=H⋅W.
We compute SVD of the row-centered data Xc=X−1μ⊤:
Xc=UΣV⊤.
Scores along the first two POD coordinates are s1=U:,1Σ11, s2=U:,2Σ22.
We define a blended ordering score
s=α s1+(1−α) s2,α=blend∈[0,1],
and sort snapshots by ascending s. This gives a smooth path through the dataset without generating new data.
Train/test split and mean removal
Denote the column-stacked ordered sequence as Xcols∈RF×T with T=N. We split columns into train (first Ttrain) and test (remaining Ttest). We remove the temporal mean of the train split:
xˉ=1Ttrain∑k=1Ttrainxk,Xtrain0=Xtrain−xˉ,Xtest0=Xtest−xˉ.
Mean-removal prevents the constant offset from being misrepresented as growing/decaying modes.

# 2. DMD Theory and the Exact Algorithm
Given consecutive pairs X=[x1,…,xT−1] and X′=[x2,…,xT] in centered space, DMD seeks a linear operator Asuch that xk+1≈Axk. Directly, A=X′X+ is huge; Exact DMD proceeds in reduced coordinates:
Reduced SVD of X: X=UΣV⊤. Truncate to rank r: Ur,Σr,Vr.
Reduced operator:
A~=Ur⊤X′VrΣr−1∈Rr×r.
Eigen-decomposition: A~W=WΛ.
Exact DMD modes (full space):
Φ=X′VrΣr−1W∈RF×r.
Amplitudes (numerically stable):
We work in reduced coordinates y=Ur⊤x. For the initial state x0, let y0=Ur⊤x0. Solve
W b0=y0⇒b0=W−1y0.
This is more stable than b0=Φ†x0.
k-step state:
x^k=ΦΛkb0,k≥0.
For an arbitrary starting state xinit, use binit=W−1(Ur⊤xinit).
Interpretation.
Columns of Φ are coherent spatial structures (DMD modes). Diagonal entries of Λ are discrete-time eigenvalues: magnitude ∣λ∣ encodes growth/decay per step; angle arg⁡λ encodes rotation/oscillation.

# 3. Evaluation Metrics
All errors are computed in the centered space (mean added back only for visualization).
Representational RMSE:
RMSErep=∥Xtrain0−Φ(Φ†Xtrain0)∥RMS.
Measures how well the mode subspace spans the data (no dynamics).
One-step RMSE:
Build A=UrA~Ur⊤ and evaluate
RMSE1-step=∥Xtrain′0−A Xtrain0∥RMS.
Measures linearity of the pseudo-time transitions.
Long-horizon train-end RMSE:
Forecast from the first training state to the last:
x^Ttrain−1=ΦΛTtrain−1b0, compare with xTtrain−1.
Test mean RMSE vs baseline:
From the last train state, forecast Ttest steps and average the RMSE across the horizon; compare with a trivial copy-last baseline.
We also visualize:
Eigenvalues on the complex plane (unit circle overlay).
Forecast pairs (True vs DMD Pred at t+1, t+2) with a shared color scale.
Train-end one-step pair (True vs Axt−1).
Train-end projection pair (True vs ΦΦ†xt)—shows subspace fit without dynamics.
Train-end long-horizon pair (True vs ΦΛKb0)—the hardest case.

# 4. Experimental Settings
Ordering: blend = 0.8 (80% POD-1 + 20% POD-2), smoother than POD-1 alone.
Rank: r = 12 (fixed).
Split: train_frac = 0.8 → Ttrain=48, Ttest=12.
Grid: 128×128.
Field: p/ρ.
Command:
python lab2_dmd_airfrans.py --root AirfRANS --var p --M 60 --H 128 --W 128 \
  --train_frac 0.8 --rank 12 --blend 0.8


# 5. Results
# 5.1 Scalars
Representational RMSE: 6.0023
→ The subspace spanned by Φ captures the training fields extremely well.
(Using the visual colorbar span ≈ 1500–1600, this is ≲ 0.5% NRMSE.)
One-step RMSE: 65.4505
→ Each step along the pseudo-time path is approximated reasonably (≈ 4% NRMSE). Expect small errors to accumulate over many steps.
Long-horizon train-end RMSE: 378.2632
→ Forecasting ~47 steps from the first frame to the last accumulates error (≈ 20–25% NRMSE), so the final panel can look different even when the subspace and one-step map are good.
Test mean RMSE (12 steps): 180.1185 vs 185.2315 (copy-last)
→ DMD beats the naïve baseline on a non-physical sequence—good sign that the learned linear map encodes some consistent drift.
# 5.2 Eigenvalues (discrete-time)
The scatter lies inside/near the unit circle, with one eigenvalue close to 1+0i.
∣λ∣≲1 → neutral/decaying dynamics per step (no explosive growth).
A point near 1 indicates slow drift of a dominant pattern along the path.
Small imaginary parts indicate weak oscillatory components (phase rotation).
# 5.3 Visuals
(All plots use a shared color scale per pair.)
Forecast t+1 and t+2: qualitative match is strong—suction peak location/intensity and recovery pattern are reproduced.
Train-end one-step (A xt−1 vs True): close agreement—consistent with the one-step RMSE figure.
Train-end projection (ΦΦ†xt vs True): near-identical—explains the very small representational RMSE.
Train-end long-horizon (ΦΛKb0 vs True): visibly different in some regions (expected cumulative drift).

# 6. Discussion
# 6.1 What the numbers mean
The subspace is excellent (RMSE ≈ 6): a few modes describe the family of fields with high fidelity.
The local linear map is reasonable (65), but global rollouts over ~50 steps amplify small biases and phase errors (378). This is typical for DMD on pseudo-time (not true dynamics).
Generalization: beating the copy-last baseline on the test horizon indicates the learned map captures a consistent progression along the POD manifold.
# 6.2 Why the long-horizon train-end plot looks different
It is not a projection; it is a many-step forecast from the first frame. Even tiny one-step errors, or an eigenvalue slightly off the unit circle, cause drift over dozens of steps. The projection and one-step visuals confirm the core model is sound.
# 6.3 Limitations
Pseudo-time ≠ physics: ordering by POD score is a surrogate for time; linear dynamics are only an approximation.
Error accumulation: unavoidable in long rollouts unless we (i) regularize, (ii) use time-delay embeddings, or (iii) perform test-time nudging with sparse observations.

# 7. Implementation Notes (concept → code)
Rasterization & mean-fill: build fixed grid, griddata(..., method="linear") with nearest fallback.
Ordering: SVD on row-centered snapshots; compute scores s=αs1+(1−α)s2; --blend controls α.
Mean removal: subtract train temporal mean; add back only for plots.
Exact DMD: SVD of X, reduced operator A~, eigen-decomp, Exact modes Φ.
Amplitudes: b0=W−1Ur⊤x0 (stable).
Forecast: xk=ΦΛkb0 (or from xinit via binit).
Diagnostics:
Representational: Φ(Φ†Xtrain0).
One-step: A=UrA~Ur⊤, compare AX to X′.
Visualization: paired plots with shared vmin/vmax to avoid misleading polarity/scale differences.

# 8. Conclusions
DMD on AirfRANS with a POD-based pseudo-time ordering yields a compact, interpretable linear evolution model. The mode subspace explains fields with sub-percent error; the linear transition model is accurate per step and outperforms a trivial baseline on held-out steps. Long-horizon deviations are expected in pseudo-time and are diagnosed (not concealed) by our four metrics and visuals.

# 9. Next Steps (low-effort, high-value)
Slightly larger rank (e.g., 10–16) and/or smoother blend (--blend 0.7–0.9) to reduce curvature error.
Ridge-regularized A~ (tiny Tikhonov) to shrink eigenvalues toward the unit circle and reduce drift.
Time-delay (Hankel) DMD to capture mild nonlinearity without neural nets.
Sensor nudging (Lab 3): at test time, apply a few gradient/least-squares steps to force the forecast to match K sensors exactly—this bridges to your sparse-sensor digital-twin project.

# 10. Reproducibility
Command:
python lab2_dmd_airfrans.py --root AirfRANS --var p --M 60 --H 128 --W 128 \
  --train_frac 0.8 --rank 12 --blend 0.8

# Artifacts:
dmd_eigs.png — eigenvalues on unit circle.
forecast_pair_0.png, forecast_pair_1.png — short-horizon forecasts.
train_last_onestep_pair.png — one-step at train end.
train_last_projection_pair.png — projection (no dynamics).
train_last_pair.png — long-horizon train end (diagnostic).
Console prints of the four metrics.



