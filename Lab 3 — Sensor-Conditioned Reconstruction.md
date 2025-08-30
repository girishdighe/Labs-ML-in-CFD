Lab 3 — Sensor-Conditioned Reconstruction and DMD Nudging on AirfRANS
Abstract
This lab addresses the classical state–estimation problem for aerodynamic fields: how to reconstruct a high-dimensional pressure field over an airfoil from a small set of sensors, and how to improve that estimate using a dynamical prior learned from data. We first learn a Proper Orthogonal Decomposition (POD) basis from training snapshots and solve a regularized least-squares problem to recover the full field from sparse measurements. We then incorporate a Dynamic Mode Decomposition (DMD) forecast as a Bayesian prior and solve a Maximum-A-Posteriori (MAP) estimator (a.k.a. nudging). On AirfRANS pressure fields rasterized to 128×128, with a POD basis of rank r=8 capturing 99.72% energy, and a DMD model of rank 12, we evaluate reconstructions of the first test frame using K=50 randomly placed, noiseless sensors. The POD-only reconstruction achieves RMSE = 29.50 (≈2% normalized error given the field’s dynamic range); the nudged reconstruction with β=1 degrades to RMSE = 86.48because the DMD prior is biased relative to the true next state while sensors are already plentiful and noise-free. An error–versus–sensor curve confirms theory: in the well-determined, low-noise regime, the optimal MAP choice is β ⁣→ ⁣0 (trust sensors); nudging becomes advantageous when sensors are few or noisy.

1. Introduction
Modern CFD workflows increasingly rely on reduced-order models and data-driven surrogates to make fast predictions and enable real-time monitoring. A central operation in such pipelines is state estimation from sparse observations: given a handful of pressure probes or strain gauges, infer the full flow field on a grid. This lab implements two principled estimators that operate entirely on open data without generating new simulations:
POD sensor reconstruction. Learn a low-dimensional subspace Φ that spans the dominant spatial structures; infer the subspace coefficients from sensor readings by solving a small linear system.
POD + DMD nudging (MAP). When a dynamical prior is available (here, a one-step forecast from DMD), combine that prior with sensors in a Bayesian least-squares objective.
Both estimators are transparent, differentiable, and easy to port across domains—attributes hiring teams at hardware and systems companies value.

2. Data and Preprocessing
2.1 Dataset and variable
We use the AirfRANS training split (800 cases). Each case contains unstructured points with coordinates (x,y) and several flow quantities. We work with p/ρ (static pressure divided by density).
2.2 Rasterization
DMD and POD require a fixed state vector. For each case we interpolate (x,y)↦p/ρ onto a Cartesian grid of size H×W=128×128 over [−2,4]×[−1.5,1.5] using linear interpolation with nearest-neighbor fill for holes. Stacking the grid values row-major yields a vector x∈RF with F=H⋅W.
2.3 Pseudo-time ordering
AirfRANS snapshots are not a time series. To exercise DMD, we construct a pseudo-time path by applying SVD to the row-centered snapshot matrix and sorting snapshots by a blend of the first two POD scores: s=α s1+(1−α) s2 with α=0.8. This creates a smooth progression of fields without synthesizing data. We then split the ordered sequence into train (first 80%) and test (remaining 20%).

3. Methods
3.1 Measurement model and POD basis
Let Xtrain∈RF×Ttrain be the column-stacked training snapshots. The POD basis is obtained from the SVD of the row-centered training matrix: the first r right singular vectors form Φ∈RF×r (orthonormal spatial modes), and the train temporal mean is μ∈RF. Any centered field is approximated as
x0≈Φa,x0≔x−μ.
Sensors measure entries of x at an index set S⊂{1,…,F} with ∣S∣=K. If S∈{0,1}K×F selects those entries, then y=Sx=Sμ+SΦa+ε. After centering, we work with
y0≔y−Sμ=Aa+ε,A≔SΦ∈RK×r.
3.2 POD-only reconstruction (regularized least squares)
With Gaussian sensor noise ε∼N(0,σ2I), the maximum-likelihood estimate of a solves
a^=arg⁡min⁡a∥Aa−y0∥22+γ∥a∥22,
whose normal equations yield the Tikhonov solution
  a^=(A⊤A+γI)−1A⊤y0  ,x^=μ+Φa^.
When K≥r and sensors are noiseless, γ can be near zero; the solution is close to exact.
3.3 DMD prior (Exact DMD)
To obtain a dynamical prior for the first test frame, we fit Exact DMD on the mean-removed training sequence Xtrain0=Xtrain−xˉ1⊤ (temporal mean xˉ). Denote X=[x1,…,xT−1] and X′=[x2,…,xT]. With the reduced SVD X=UΣV⊤ truncated to rank rd, the reduced operator is
A~=Ur⊤X′VrΣr−1.
Eigen-decomposition A~W=WΛ yields DMD modes
ΦDMD=X′VrΣr−1W
and a k-step prediction in the centered space
x^k=ΦDMDΛkb0,b0=W−1(Ur⊤x0).
For the first test step, we forecast one step ahead from the last train snapshot, convert to absolute space by adding xˉ, and project onto the POD basis to obtain a prior coefficient vector a0:
a0=arg⁡min⁡a∥Φa−(x^abs−μ)∥22  =  (Φ⊤Φ)−1Φ⊤(x^abs−μ)  =  Φ⊤(x^abs−μ),
since Φ has orthonormal columns.
3.4 Nudging as MAP estimation
With prior a∼N(a0,τ2I) and sensor noise ε∼N(0,σ2I), the MAP estimator solves
a^=arg⁡min⁡a∥Aa−y0∥22⏟sensor fit+β∥a−a0∥22⏟prior pull+γ∥a∥22,with β=σ2/τ2.
The solution is closed form:
  a^=(A⊤A+(β+γ)I)−1(A⊤y0+βa0)  ,x^=μ+Φa^.
Thus β controls the trade-off between matching sensors and staying close to the prior.

4. Experimental Protocol
We ordered N=60 rasters with blend α=0.8, used Ttrain=48 and Ttest=12.
The POD basis kept r=8 modes (energy 99.72%).
The DMD model used rank 12; its eigenvalues lie mostly inside or near the unit circle with one close to 1+0i, indicating slow, neutral drift.
We evaluated the first test step (the snapshot immediately after the train horizon). Sensors were random pixels(uniform over the grid). For the run under discussion:
Number of sensors K=50,
Sensor noise σ=0 (noiseless),
Regularization γ=10−6 (stability only),
Nudging weight β=1.
All reconstructions are reported as full-field RMSE against the truth (temporal mean added back), and paired images are rendered with a shared color scale to allow visual comparison.

5. Results
5.1 Scalars
The POD-only estimator achieved RMSE = 29.50. The color scale in the figures spans roughly 1.5×103 units, so this error corresponds to about 2% normalized RMSE, which is excellent for a single-frame reconstruction from sparse point samples.
The nudged estimator with β=1.0 yielded RMSE = 86.48. This is worse than POD-only, not because the code is wrong, but because a non-zero β deliberately pulls the solution toward the DMD prior a0. In our setting the prior carries a small but systematic bias (DMD is trained on a pseudo-time ordering rather than true dynamics). When sensors are both numerous and clean, the statistically optimal choice is β→0; any prior pull adds error.
5.2 Visual inspection
The POD-only reconstruction closely matches the true suction pocket near the leading edge, the recovery ridge downstream, and the off-body gradients. The nudged reconstruction looks slightly “softened” and shifted in those regions—exactly the effect of blending in a biased prior when you don’t need it.
The DMD eigenvalue plot shows a constellation well inside the unit circle with a point near (1,0); this spectrum encodes slow drift and mild damping, which is consistent with Lab 2’s observation that long-horizon DMD rollouts on pseudo-time accumulate small phase and amplitude errors.
5.3 Error versus number of sensors
We also swept K∈{5,10,20,50,100,200} at fixed β=1 and no noise. The POD-only curve drops sharply as Kincreases and plateaus in the low-20s by K≈100. The nudged curve, by contrast, remains almost flat around ≈86, demonstrating that with no noise the error is dominated by the prior bias introduced by β>0; adding sensors does not remove that bias.

6. Interpretation
The three ingredients—representation, measurement, and prior—explain the numbers:
Representation. The POD basis captures 99.72% of train variance with r=8. In practice, the test field lies very close to that subspace, so if you knew the true coefficients a⋆, ∥Φa⋆−x0∥ would be tiny. This is exactly why the POD-only solver performs so well with enough sensors.
Measurement. With K=50 and r=8, the system Aa≈y0 is well-overdetermined and noiseless. The least-squares coefficients are therefore very accurate; adding γ only to stabilize conditioning has negligible effect.
Prior. The DMD prior is a forecast sitting slightly off the true target due to (i) pseudo-time not matching physical dynamics, and (ii) small model reduction errors. When β>0, the MAP solution moves toward this biased point; if the sensors were few (underdetermined) or noisy, this movement would reduce variance and improve RMSE. Here, with many clean sensors, it only adds bias, hence higher error.
In Bayesian language: the likelihood (sensors) is already very informative; the prior has non-zero mean error; the MAP estimate trades variance for bias you do not need.

7. Sensitivity and “When nudging helps”
Although nudging hurt in this specific, clean setup, it is powerful in the regimes practitioners care about:
Few sensors (underdetermined). If K<r or K is only slightly larger than r, A⊤A is ill-conditioned and the POD-only solution becomes unstable. A moderate β anchors the solution and typically reduces RMSE.
Noisy sensors. When σ>0, the ML solution overfits noise. The MAP weight β (heuristically β≈σ2/τ2) regularizes toward the prior and improves accuracy. In practice, try a small grid β∈{0,0.1,1,5,10} and select on a validation step.
Better priors. If the DMD prior is improved—e.g., via time-delay (Hankel) DMD, tiny ridge on A~ to shrink eigenvalues toward the unit circle, or a shallow sequence model—nudging becomes useful even with more sensors.

8. Implementation Notes (concept → code)
The script constructs the sensing matrix by selecting rows of Φ at the sensor indices: A=ΦS,:. It solves the two linear systems
(A⊤A+γI)a^=A⊤y0,(A⊤A+(β+γ)I)a^=A⊤y0+βa0
with dense linear algebra. Because r is small (single-digit to teens), these solves are trivial in cost. All comparisons and RMSEs are computed in absolute space by adding μ back to the centered reconstructions. Paired images are rendered with a shared (vmin⁡,vmax⁡) computed from the 1st–99th percentiles of the two panels, eliminating false visual “inversions.”
A subtle but important detail is the centering alignment: DMD is trained in train-mean centered coordinates, whereas POD reconstructions are expressed relative to the POD mean μ. The code first returns the DMD prediction to absolute space with the train mean, then re-centers by μ before projecting into the POD basis to form a0. This prevents mean-offset leakage into the prior.

9. Limitations and Practical Enhancements
This lab used random sensors. In applications you usually want informative sensor locations. Deterministic choices such as Q-DEIM or farthest-point sampling on the columns of Φ (or on leverage scores) place sensors near regions of high variability and markedly reduce RMSE for the same K.
Because the pseudo-time path is not physical time, DMD introduces a mild bias. If you aim to deploy nudging routinely, consider (i) time-delay DMD to approximate local nonlinearity, (ii) a small ridge in A~ to discourage growth, or (iii) a lightweight recurrent prior.
Finally, for multi-variable fields (u,v,p), standardize channels before POD so modes do not collapse onto the largest-magnitude variable.

10. Conclusions
This lab demonstrates a complete, reproducible pipeline for sensor-conditioned reconstruction of aerodynamic fields:
A rank-8 POD basis captures nearly all variance; with K=50 noiseless sensors, POD-only reconstruction achieves ≈2% NRMSE on the first test frame.
Incorporating a DMD prior with β=1 increases error to ≈86 because the prior is slightly biased and unnecessary when sensors are abundant and clean.
An error–versus–sensor study confirms the expected Bayesian behavior: in the high-information, low-noise regime you should trust sensors (β→0); nudging becomes valuable precisely when the inverse problem is ill-posed or noisy.
These findings are not only theoretically consistent; they also map directly to industrial practice: learn a compact basis once, place a handful of informative sensors, and choose β according to the noise/sensor budget. The code is small, portable, and immediately extensible to multi-physics twins.

11. Reproducibility
Command used for the main run
python lab3_sensor_twin_airfrans.py --root AirfRANS --var p --M 60 --H 128 --W 128 \
  --train_frac 0.8 --pod_rank 8 --sensors 50 --beta 1.0 --noise_std 0.0 \
  --blend 0.8 --dmd_rank 12

Artifacts produced
recon_pod_only.png — true vs POD-only reconstruction.
recon_nudged.png — true vs nudged reconstruction.
rmse_vs_sensors.png — error curves for K∈{5,10,20,50,100,200}.
dmd_eigs.png — DMD spectrum (unit circle overlay).
sensor_indices.npy — the exact sensor locations used.

Appendix A — Derivations (compact)
POD-only normal equations.
Minimize J(a)=∥Aa−y0∥22+γ∥a∥22. Setting ∇J=0 gives
(A⊤A+γI)a=A⊤y0.
MAP nudging.
With likelihood p(y0∣a)∝exp⁡(−12σ2∥Aa−y0∥22) and prior p(a)∝exp⁡(−12τ2∥a−a0∥22), the negative log posterior is (up to constants)
∥Aa−y0∥22+(σ2/τ2)∥a−a0∥22. Identifying β=σ2/τ2 and adding a small numerical ridge γ yields the nudging objective used in code and its closed-form solution.
Connection to Kalman update (static).
In the special case of Gaussian priors and linear, noisily observed states, the MAP solution above is algebraically equivalent to a single Kalman correction in the POD coordinates; β plays the role of the prior covariance inverse.


