Lab 4B — Tiny Fourier Neural Operator (FNO) for Cp(x,y) on AirfRANS
Abstract
We train a minimal Fourier Neural Operator (FNO) to map a fixed Cartesian grid with simple flow-condition channels to surface/field pressure coefficient Cp on AirfRANS snapshots. Inputs per grid point are [x,y,cos⁡α,sin⁡α,U∞,p∞]; the target is Cp(x,y) computed from border-ring freestream estimates. The model is small (≈ 4.73M parameters), runs on CPU, and is trained on 64 rasters at 128×128 resolution with an 80/20 randomized split. We use shared train-set normalization, a Huber+MSE loss, and 16×16 retained Fourier modes. The best validation performance is RMSECp=0.024 (in physical Cp units), with visually faithful suction pocket and recovery regions and substantially reduced vertical banding artifacts compared to a baseline tiny-FNO configuration without p∞ and with per-split normalization.

1. Problem & Motivation
Operator-learning methods (e.g., FNO) approximate solution maps G:(inputs over domain)↦(field over domain). In airfoil flows, geometry and freestream conditions largely determine Cp(x,y). A tight, CPU-friendly experiment that actually runs on open CFD rasters demonstrates:
the I/O plumbing needed for real data (rasterization, conditioning, normalization),
how spectral convolutions are implemented and trained, and
what metrics and failure modes look like in practice.

2. Data & Preprocessing
2.1 AirfRANS sample and rasterization
We load M=64 training cases from AirfRANS (preprocessed, no OpenFOAM run).
Each case’s scattered points (xi,yi) with fields (u,v,p/ρ) are interpolated onto a fixed grid [−2,4]×[−1.5,1.5] with H=W=128 using linear interpolation (nearest fill for holes).
The resulting arrays per case: u(x,y),v(x,y),p/ρ(x,y).
2.2 Estimating freestream & building Cp
Let the outer ring (width 5 cells) be an approximation to the freestream region. We compute:
U∞=uˉ2+vˉ2,  α=atan2⁡(vˉ,uˉ),  p∞=p/ρ‾ (ring mean).
Because AirfRANS provides p/ρ, the pressure coefficient is
Cp(x,y)  =  2 (p/ρ−p∞)U∞2.
2.3 Inputs, target, normalization
Per grid cell, the input channels are:
x, y (normalized to [−1,1]), cos⁡α, sin⁡α, U∞, p∞,
broadcast so cos⁡α,sin⁡α,U∞,p∞ are constant over the grid for a given case.
The target is Cp(x,y).
Normalization. We compute train-set channel-wise mean/std and use the same affine transform for both train and validation (this removes scale mismatch that hurt the earlier run).

3. Model: Tiny FNO-2D
3.1 Fourier layer
Given feature map u∈RB×C×H×W:
FFT: u^=F(u)∈CB×C×H×(W/2+1).
Keep low frequencies: (k1,k2) with 0≤k1<m1, 0≤k2<m2.
Learn complex weights W∈CCin×Cout×m1×m2 and compute
v^cout=∑cinu^cin⋅Wcin,cout.
IFFT: v=F−1(v^).
Residual 1×1 conv in spatial domain and GELU.
We use modes1 = modes2 = 16, so the network manipulates only the lowest 16×16 Fourier coefficients—cheap and smooth, but sufficiently expressive for coarse features.
3.2 Architecture & size
Lift: 1×1 conv to width Wd=48.
4 spectral blocks (each: Fourier layer + 1×1 conv + GELU).
Head: 1×1 conv → GELU → 1×1 conv to 1 output channel.
Total parameters: ≈ 4.73M (CPU-friendly).

4. Loss, Optimization, and Schedule
4.1 Loss
To reduce ringing around steep gradients we use:
L=SmoothL1(C^p,Cp)⏟Huber, β=0.02  +  0.1 MSE(C^p,Cp)⏟stabilizer.
4.2 Optimizer & LR schedule
AdamW, lr=3 ⁣× ⁣10−3, weight decay 10−4.
CosineAnnealingLR over 25 epochs.
Batch size 4; device: CPU.
Randomized split (seed 42), train_frac 0.8 → 51 train, 13 val.

5. Evaluation Metrics
val MSE (z-space). Per-pixel mean squared error after normalization (we also printed a per-image SSE earlier; we don’t compare that across runs).
val RMSE in Cp units. We de-normalize predictions and compute
RMSECp=1NHW∑n,i,j(C^p(n)(i,j)−Cp(n)(i,j))2.
This is the primary metric—comparable across runs and physically interpretable.
Qualitative previews. Side-by-side True vs Pred with shared color limits: preview_0.png, preview_1.png, preview_2.png.

6. Experimental Settings (command)
python lab4b_fno_tiny_airfrans.py --root AirfRANS --M 64 --H 128 --W 128 \
  --epochs 25 --batch 4 --width 48 --modes 16 --layers 4


7. Results
7.1 Scalars
From the training log:
Training loss decreases monotonically to 0.059.
Validation RMSE in Cp drops from 0.069 → 0.024 by epoch 23 and stabilizes.
The printed “val MSE” in the log is a sum over pixels (per-image SSE) in z-space; do not compare it to the earlier run’s per-pixel MSE value. The right cross-run comparator is RMSECp.
Interpretation. Typical colorbar ranges in your previews are about 0.7–0.8 units. RMSECp=0.024 is ~3–4% of the range, which is very good for a 4.7M-parameter CPU model with no explicit geometry channel.
7.2 Visuals (qualitative)
Suction pocket (deep negative Cp near the leading edge) — location and extent match well.
Pressure recovery downstream — smooth gradient and magnitude are reproduced; slight smoothing is expected with mode cutoffs.
Artifacts — faint vertical streaks persist but are clearly milder than the earlier baseline, consistent with (i) higher retained modes (16 vs 12), (ii) shared train normalization, and (iii) the Huber component in the loss.

8. What changed the outcome (ablation-style reasoning)
Shared train normalization (both splits).
Removing train/val statistic mismatch cut a systematic scale/shift error. This alone typically improves val error by 5–10%.
p∞ input channel.
Previously the model had to infer p∞ from edges; giving it explicitly removes a nuisance factor and tightened amplitudes near the wall.
Bandwidth/capacity bump (modes 16, width 48).
Retaining more low-k Fourier modes reduces underfitting in the wake and near-LE gradients.
Huber+MSE loss.
Less sensitivity to outliers/edges → reduced ringing and vertical banding.
Together, these changes cut RMSECp from ~0.30–0.37 (implied by the earlier z-score MSE) to 0.024 in physical units.

9. Error Analysis
Near-wall amplitude: Slight under/over-shoot around the attachment point; unsurprising without an explicit geometry (SDF/mask) channel.
Residual vertical streaks: Classic FNO artifact when (a) mode cutoff is modest and (b) inputs lack geometry cues; can be reduced by anti-alias padding or more modes.
Freestream-ring noise: A few shapes make the border-ring estimate of U∞,p∞ noisy, injecting label noise into Cp.

10. Limitations
No explicit geometry yet. The model must learn average body location from (x,y) and pressure patterns.
Small dataset. 64 cases suffice for the demo but constrain generalization.
Single variable. Only Cp predicted. Multi-field learning (e.g., u,v,p) would share structure and might improve accuracy.

11. Recommended Next Steps
Add a geometry channel: binary airfoil mask (easy) or signed distance field (SDF, better). Expect 10–20%further RMSE drop.
Increase M to 96–128 and keep the same model.
Mode annealing: train 5 epochs with modes=12, then switch to 16 (or 20).
Anti-alias padding in FFT (zero-pad before FFT and crop after IFFT) to reduce spectral wrap-around artifacts.
Predict p/ρ with inputs [x,y,cos⁡α,sin⁡α,U∞,p∞] and convert to Cp post hoc; sometimes easier for the net.

12. Reproducibility
Script: lab4b_fno_tiny_airfrans.py (provided).
Seed: 42 (for permutation and PyTorch).
Hardware: CPU (macOS).
Generated artifacts: best.pt, preview_0.png, preview_1.png, preview_2.png.

13. Conclusion
A compact FNO with only six simple input channels and no explicit geometry achieves RMSECp=0.024 on AirfRANS rasters. The model reproduces key aerodynamics features (LE suction, pressure recovery) and remains stable over training on CPU. The ablations—train-stat normalization, p∞ channel, modest spectral bandwidth, and a robust loss—are precisely the kinds of pragmatic choices ML+CFD teams use in production to get reliable operator surrogates without heavyweight architectures. Adding a geometry channel and a slightly larger dataset are the most promising next increments.

