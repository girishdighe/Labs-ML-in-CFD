# Lab 1 — Proper Orthogonal Decomposition (POD) of Airfoil Pressure Fields (AirfRANS)

# Abstract
This lab demonstrates how to construct a low-dimensional linear basis for 2-D airfoil pressure fields using Proper Orthogonal Decomposition (POD) via the Singular Value Decomposition (SVD). We (i) rasterize p/ρ from the AirfRANS dataset onto a fixed grid, (ii) form a snapshot matrix, (iii) compute the SVD of the mean-removed data, (iv) select a rank r by cumulative energy, and (v) evaluate reconstruction error. On 50 snapshots (128×128), five POD modes capture 99.33% of the variance with RMSE ≈ 14.63 (in the plotting units of p/ρ), producing visually faithful reconstructions. The first three modes align with physically interpretable structures: (1) leading-edge suction vs lower-surface compression (lift-like dipole), (2) adjustments to suction peak location/intensity, and (3) downstream pressure-recovery/wake variability.

# 1. Introduction
High-fidelity CFD fields are high-dimensional. Many configurations, however, live near a low-dimensional manifold: a small set of spatial patterns explains most variability across cases, operating points, or geometries. POD provides the optimal rank-r linear approximation (in the least-squares sense) by projecting data onto orthonormal spatial modes learned from the data itself. This lab builds the full POD workflow end-to-end and ties each mathematical step to the corresponding code.
Goal. Build and validate a compact basis for pressure fields over airfoils to (a) compress fields; (b) prepare a latent space that later ML models can predict (ROMs), and (c) lay the groundwork for sensor-conditioned reconstructions (digital twins).

# 2. Dataset and Preprocessing
# 2.1 AirfRANS subset
We use the AirfRANS “train/full” split and select M=50 simulations to keep the SVD inexpensive. Each simulation returns node-wise arrays with features and targets. We extract the target p/ρ (static pressure divided by density), because it is smooth and physically meaningful.
# 2.2 Domain and rasterization
AirfRANS provides unstructured point clouds (mesh nodes). POD requires a consistent feature ordering per snapshot. We therefore define a fixed Cartesian grid:
x∈[−2, 4],y∈[−1.5, 1.5],H=W=128,
and interpolate node values to that grid using linear interpolation; any empty pixels are filled with nearest-neighborvalues. This yields a 3-D tensor X∈RN×H×W, here N=50.
Implementation (key ideas).
scipy.interpolate.griddata with method="linear" distributes nearby node values onto grid points, preserving smoothness.
A nearest-neighbor fallback fills holes that arise near sharp gradients or sparsely sampled regions.
Assumptions.
The grid bounds cover the flow features for all cases (no clipping).
Interpolation smooths sub-grid features slightly; this can reduce the effective rank and is acceptable for a first POD.

# 3. Theory
# 3.1 Snapshot matrix and centering
Flatten each raster to a row vector xi∈RF with F=H⋅W (or H⋅W⋅C if multi-channel). Stack rows to obtain
X=[x1⊤⋮xN⊤]∈RN×F.
Compute the feature-wise mean μ=1N∑i=1Nxi, then center:
X~=X−1μ⊤.
Centering removes the mean field; POD modes then represent fluctuations around the mean.
# 3.2 SVD and POD modes
Compute the compact SVD:
X~=U Σ V⊤,
where U \in \mathbb{R}^{N\times r_\max}, V \in \mathbb{R}^{F\times r_\max}, \Sigma = \mathrm{diag}(\sigma_1,\ldots,\sigma_{r_\max}), and σ1≥σ2≥⋯. The POD spatial modes are columns of V (equivalently, rows of V⊤); the modal coefficients for snapshot i are the i-th row of UΣ.
Optimality. By the Eckart–Young theorem, the rank-r approximation
X~r=UrΣrVr⊤
minimizes ∥X~−Y∥F over all matrices Y of rank at most r.
# 3.3 Energy and rank selection
Define the cumulative energy ratio:
\mathrm{CE}(r) = \frac{\sum_{k=1}^r \sigma_k^2}{\sum_{k=1}^{r_\max} \sigma_k^2}.
We choose the smallest r such that CE(r)≥τ, with τ=0.99 in this lab.
# 3.4 Reconstruction and error metrics
Reconstruct each centered snapshot by
x^i=μ+∑k=1raik ϕk,
where ϕk is mode k (column k of V) and aik is the k-th coefficient of snapshot i (from UΣ). We report the root-mean-square error:
RMSE=1NF∑i=1N∥xi−x^i∥22.
Notes.
POD modes are defined up to sign (flipping a mode and its coefficients leaves the product unchanged).
Orthonormality: modes form an orthonormal basis in feature space, i.e., V⊤V=I.

# 4. Methods (Algorithm → Code)
# 4.1 End-to-end steps
Load & rasterize AirfRANS (Sec. 2.2) → tensor X∈RN×H×W.
Flatten: reshape to X∈RN×F.
Center by column mean μ.
SVD of X~ (compact).
Select r by CE(r)≥0.99.
Reconstruct X^ using UrΣrVr⊤+μ.
Evaluate RMSE; plot scree, energy curve, modes, original vs. reconstruction.
# 4.2 Practical considerations
With N≪F (50 snapshots vs. 16,384 features), compact SVD is efficient.
Interpolation can introduce small NaN islands; nearest-neighbor fill avoids artifacts in SVD.
If N grows large, randomized SVD is a drop-in accelerator.

# 5. Experimental Setup
Snapshots: N=50 AirfRANS cases (train/full split).
Field: p/ρ.
Grid: 128×128 over [−2,4]×[−1.5,1.5].
Rank rule: smallest r with CE(r)≥0.99.
Metrics: RMSE over all pixels and snapshots; qualitative inspection of modes and reconstructions.
Command used:
python lab1_airfrans_pod.py \
  --root AirfRANS \
  --var p \
  --M 50 \
  --H 128 --W 128 \
  --energy 0.99


# 6. Results
# 6.1 Energy curve (cumulative energy)
The curve rises sharply and crosses 99% at r=5, then quickly saturates toward 100%.
Interpretation: the dataset’s variability is highly low-rank; a handful of spatial patterns explains almost all variance.
# 6.2 Scree plot (singular values)
A steep decay in σk over the first few modes, followed by a long tail.
Physical reading: one dominant mechanism, one or two secondary mechanisms, then minor corrections—typical of quasi-potential pressure distributions around similar shapes/AoA.
# 6.3 Spatial POD modes (0–2)
(modes visualize fluctuations relative to the mean)
Mode 0 (most energetic). Dipole centered slightly aft of the leading edge: strong suction (negative) above and compression (positive) below. This captures the lift-like skew across cases (AoA/shape differences).
Mode 1. A subtler pattern adjusting the location and amplitude of the suction peak, with side lobes indicating spanwise-symmetric corrections to Mode 0.
Mode 2. Streamwise lobe extending downstream, aligned with pressure recovery / wake variations: how quickly the field returns toward freestream behind the trailing edge.
These patterns match aerodynamic intuition: the leading-edge region dominates pressure variability; next come shifts in peak position/intensity; finally wake-recovery differences.
# 6.4 Reconstructions (original vs rank-5)
Visual agreement is excellent. Differences are confined to regions of steep gradients (near the leading edge and immediately aft of the airfoil), which naturally require additional modes to reproduce perfectly.
Reported metrics:
Chosen rank: r=5
Energy captured: 99.33%
RMSE: 14.6299 (units of p/ρ as plotted).
Given the colorbar range seen in the figures (approximately −1.8×10³ to +0.6×10³), the normalized erroris on the order of 0.5–1%, consistent with the high energy capture.

# 7. Discussion
# 7.1 Are five modes “enough”?
For compression and as a latent space for downstream learning, yes. r=5 balances compactness and fidelity. If your downstream task requires sub-percent accuracy at the sharpest gradients, increasing to r=8–12 will reduce residuals near the leading edge.
# 7.2 Physical interpretability
POD does not “know” physics, yet the dominant modes align with lift distribution, suction peak modulation, and wake recovery—precisely the mechanisms that vary across airfoil shapes and AoA/Re. This interpretability is valuable when you later predict modal coefficients with a neural network: you can relate coefficient shifts to physical effects.
# 7.3 Limitations
Interpolation smoothing. Linear interpolation plus nearest fill slightly damps very fine scales; a conservative or higher-order remap would better preserve small features (at higher cost).
Single variable. Using only p/ρ keeps the rank low. Joint POD over [u,v,p] typically requires a larger r.
Global basis. A single basis over diverse cases is powerful but may blur niche behaviors; local or clusteredPODs can tighten error if needed.
# 7.4 What this unlocks next
ROM for geometry generalization. Learn f(shape SDF,α,Re)↦a1:r (coefficients), reconstruct with fixed modes Vr.
Sensor-conditioned reconstruction. At test time, solve a tiny least-squares (or few gradient steps) on a to match sparse sensors exactly while staying on the POD manifold.

# 8. Reproducibility
Dependencies
Python ≥3.10, NumPy, SciPy, Matplotlib, AirfRANS (and its dependencies).
Determinism
SVD is deterministic for fixed data; interpolation produces identical rasters for the same inputs and grid.
How to replicate
Install requirements.
Run the command in §5.
Inspect outputs in pod_outputs/: scree.png, energy_cumulative.png, mode_0.png.., recon_pair_*.png.
Change --var umag to repeat on velocity magnitude.
Adjust --M, --H, --W to test sensitivity.

# 9. Conclusion
This lab produced a compact, interpretable basis for airfoil pressure fields: five POD modes recover >99% of variance with low reconstruction error. The spatial structures match aerodynamic intuition and provide a practical latent space for ML-accelerated reduced-order modeling and sensor-driven digital twins.

# Appendix A — Concepts Mapped to Code (quick index)
Rasterization & NaN fill → load_airfrans_as_rasters(...)
Snapshot matrix & flattening → X.reshape(N, H*W*1)
Centering → mean = X_flat.mean(...); Xc = X_flat - mean
SVD (economy) → np.linalg.svd(Xc, full_matrices=False)
Energy & rank selection → cumulative sum of S**2; choose smallest r ≥ threshold
Reconstruction → (Ur * Sr) @ VTr + mean
RMSE → rmse(Xrec, X)
Plots → scree, energy curve, first modes, original vs. reconstruction

# Appendix B — Practical Tips
Scaling: for multi-variable POD, standardize channels to comparable scales.
Randomized SVD: use for very large N (e.g., sklearn.utils.extmath.randomized_svd).
Mode sign: free to flip; focus on patterns, not sign.
Choosing r: report both CE(r) and field RMSE; for ROMs, also validate downstream performance (e.g., lift/drag deviations) vs. r.




