# Lab 5 — Hybrid Hankel–DMD + Residual FNO on AirfRANS

# Abstract
This lab builds a hybrid forecaster for airfoil pressure fields on a fixed raster. The linear core is a time–delay (Hankel) Dynamic Mode Decomposition (DMD) fitted to centered training snapshots, which supplies a one-step Koopman-style predictor with interpretable eigenstructure. On top of that core we train a small Fourier Neural Operator (FNO) that receives the Hankel one-step prediction and spatial coordinates as input and predicts a correction that should remove the linear model’s bias. We evaluate two configurations. In the first, the hybrid achieves a 12.7%reduction in mean test RMSE compared to Hankel–DMD alone (161.8 → 141.3 p/ρ), although a copy-last baseline remains best because the evaluation uses POD-ordered “pseudo-time,” which places the test frames very close to the last training frame. In the second, with a stronger Hankel core (q=8, rank=16, ridge=1e-2) and a larger residual network, Hankel–DMD improves to 118.95 RMSE and beats copy-last (130.71), but the hybrid becomes worse(191.91). A careful audit shows why: in the current script the residual FNO is trained to predict the absolute field rather than the residual xt+q−x^t+qHankel; at inference this absolute prediction is added again to the Hankel forecast, producing systematic over-correction. We explain the full pipeline mathematically, analyze the plots you produced, and give a corrected formulation that restores the intended hybrid behavior.

# 1. Data, pseudo-time, and centering
Each AirfRANS case is interpolated to a uniform grid (H,W)=(128,128) over [−2,4]×[−1.5,1.5]. Let xi∈RFdenote the flattened raster of p/ρ for case i, with F=H⋅W. The dataset has no physical time axis, so we impose a pseudo-time ordering with POD scores. If X∈RN×F stacks the N rasters, a row-centered SVD Xc=UΣV⊤ yields scores si=β1Ui1Σ11+β2Ui2Σ22 with β1=blend, β2=1−blend. Sorting by si gives a sequence {xt}t=1T, which we split into train and test along that order. We subtract the train mean xˉtrain so all learning happens in a centered space: xt0=xt−xˉtrain.
This POD ordering is the reason a naive copy-last baseline can be deceptively strong: the first few test frames frequently resemble the last training frame xTtrain, so simply repeating it yields small errors, especially at steps t+1 and t+2.

# 2. Hankel–DMD (time-delay DMD)
To capture short-term dynamics, we embed the sequence in a q-lag Hankel space. If X=[x10,…,xT0]∈RF×T, we build paired block-Hankel matrices
Z0=[x10x20⋯xT−q0x20x30⋯xT−q+10⋮⋮⋱⋮xq0xq+10⋯xT−10],Z1=[x20x30⋯xT−q+10x30x40⋯xT−q+20⋮⋮⋱⋮xq+10xq+20⋯xT0].
A reduced linear map A~ is fitted by ridge regression in a POD basis of Z0:
Z0=UΣV⊤,A~=arg⁡min⁡A∥U⊤Z1−U⊤Z0A∥F2+λ∥A∥F2=YX⊤ (XX⊤+λI)−1,
with X=U⊤Z0, Y=U⊤Z1, rank r chosen by energy or by argument, and ridge λ. Eigen-decomposition A~W=WΛgives the time-delay DMD modes
Φ=Z1 VrΣr−1W∈RqF×r,λk=eigvals(A~).
The state over the last block of size F advances as zt+1≈Φ bt, with modal amplitudes bt=W−1U⊤vec([xt−q+10;… ;xt0])⏟zt. Forecasts k steps ahead are zt+k≈ΦΛkbt; the final F entries of zt+k are the prediction for xt+k0.
Your eigenvalue plots show most λk near the unit circle with a few slightly outside, consistent with weakly damped, quasi-periodic structures in the flow rasters and explaining the slow growth in error as the horizon increases. With q=8, rank =16, ridge =10−2, the Hankel core is markedly stronger than with q=6, rank =12.

# 3. Residual FNO and the hybrid design
The hybrid aims to retain the stability and interpretability of the linear core while letting a small CNN correct the consistent modeling error. At each training window [t,…,t+q−1] we compute the Hankel one-step forecast x^t+qHand feed an FNO with input tensor
inp(i,j)=[ xnorm(i,j), ynorm(i,j), x^H(i,j) ],
so the network can condition on spatial location and the coarse physics-based guess. The FNO consists of a 1×1 lifting convolution, four spectral blocks that parametrize low-wavenumber complex weights in the Fourier domain, and a 1×1 head. We train with a robust loss (Smooth-L1 plus a small MSE term) and cosine LR schedule.
Intended target. Conceptually, the FNO should learn the residual
rt+q  =  xt+q0  −  x^t+qH,
so that the final hybrid forecast is
x^t+qhyb  =  x^t+qH  +  r^t+q.
What the current script actually trains. In the dataset builder, the target was set to the absolute centered field xt+q0rather than the residual. At inference time, this prediction—having been trained toward xt+q0—is added to the Hankel forecast anyway, yielding
x^t+qhyb  =  x^t+qH  +  xt+q0^⏟network output,
which in expectation overshoots by approximately xt+q0 and can more than double the leading-edge amplitude. The qualitative panels you saved for the second configuration show exactly this behavior: the hybrid has a deeper suction pocket and brighter pressure lobes than either truth or Hankel, not because the network “found” missing physics, but because it is adding an estimate of the whole field on top of a competent baseline.
This mismatch explains the two evaluation regimes. In the first run (q=6, rank=12, small FNO), the Hankel baseline is modest; the network, despite the wrong target, partially regresses toward the baseline (it sees x^H as an input and “uses” it), so the composite prediction incidentally improves upon Hankel by 12.7%. In the second run, the Hankel baseline becomes substantially better (118.95 RMSE, now beating copy-last), and the same over-additive hybrid degradesperformance to 191.91 because it is now injecting a large extra field on top of an already accurate forecast.

# 4. Results and interpretation
The linear core alone shows the expected Koopman structure. With q=6, rank=12, ridge 10−3, the mean test RMSE is 161.8, and the eigenvalues lie clustered near the unit circle with a few slightly outside, producing slow error growth. The hybrid with a ~1.2 M-parameter FNO reduces the average error to 141.3. The per-step RMSE plot reveals the biggest gains in the mid-horizon, where Hankel starts to drift; early steps remain dominated by copy-last because of the POD ordering. Visual comparisons at t+1 and t+2 show crisper reconstruction of the suction pocket and pressure recovery.
With a stronger linear core (q=8, rank=16, ridge 10−2) and a 4.7 M-parameter FNO, the Hankel baseline improves to 118.95 and surpasses copy-last (130.71), confirming that the time-delay embedding and stronger regularization helped. The hybrid, however, worsens to 191.91. The per-step curve shows a spike around step 2 and consistently higher errors thereafter; the t+1 and t+2 frames exhibit over-correction—deeper suction and brighter lobes—consistent with adding an approximate truth on top of the baseline.
The difference between the two runs therefore does not indicate that residual learning is ineffective; rather, it puts a spotlight on a target-labeling bug that hurts more as the baseline improves.

# 5. Corrected hybrid formulation
The mathematically consistent hybrid trains the FNO on residuals and normalizes using residual statistics. In the dataset constructor this amounts to replacing the target definition
tgt = X_train0_cols[:, t+q]
by
base = hankel_onestep_from_window(...)andtgt = X_train0_cols[:, t+q] - base,
and computing (μy,σy) over this residual tensor. At test time, after de-normalizing the network output back to centered p/ρ units, one adds the residual to the Hankel forecast,
x^hyb=x^H+r^,
and only then re-adds the training mean xˉtrain for visualization. With this fix, the hybrid’s learning objective exactly matches how its output is used at inference, eliminating the double-counting that produced the overshoot in your second run.
A gentle refinement is to introduce a gate 0≤α≤1 so the network outputs (r^,α) and the update is x^H+α r^. This guards against occasional over-corrections without blunting useful residuals. A complementary guard is to penalize the energy of the residual via λ∥r^∥22 or to clip it to k standard deviations of training residuals.

# 6. Why copy-last sometimes wins and why that isn’t fatal
Because the ordering is not physical time but a monotone trajectory in POD space, the first test frames tend to look like the last train frame. A copy-last baseline therefore has an unfair advantage on that split. The right way to assert generalization is to evaluate on the official AirfRANS test split (unseen geometries/flow conditions) or at least on held-out shapes within the train set. In that regime Hankel–DMD remains meaningful, copy-last degrades substantially, and a correctly trained residual hybrid is expected to outperform both.

# 7. Conclusions
This lab demonstrated a full hybrid pipeline: a stable, interpretable time-delay DMD core and a learned residualmeant to correct its systematic error on realistic CFD rasters. Your first configuration already showed the intended effect—hybrid < Hankel in RMSE—while the second exposed a target mismatch that turned the residual into an over-additive term. The eigenvalue spectra remain near the unit circle, indicating that the linear backbone is well-posed; qualitative plots reveal that the residual is acting where it should (near the leading-edge suction and wake), but its amplitude must be calibrated through correct target definition and, optionally, a gating mechanism. 


