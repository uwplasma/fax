# Theory and numerics

This page derives every equation FAX solves, in the order the code applies
them, and names the function that implements each one. Units are SI;
frequencies `ω` are in rad/s and the public functions report `f = ω/2π` in Hz.

## 1. Boozer coordinates and the metric

FAX works on flux surfaces labelled by the normalized toroidal flux
$s \in (0, 1)$, with Boozer angles $(\theta, \zeta)$. The field is

$$
\mathbf B = \nabla\psi\times\nabla\theta + \iota\,\nabla\zeta\times\nabla\psi
          = G(s)\,\nabla\zeta + I(s)\,\nabla\theta + K\,\nabla\psi ,
$$

so the Boozer Jacobian is $\sqrt g_B = (G + \iota I)/B^2$ (up to the flux
normalization) and the parallel derivative is

$$
\mathbf B\cdot\nabla = \frac{B^2}{G + \iota I}\left(\partial_\zeta + \iota\,\partial_\theta\right).
$$ (bdotgrad)

A {class}`fax.Geometry` stores, on each surface, the Boozer harmonics of
$|B|$ (`bmnc`, `bmns`), of $R$, $Z$ and of $p = \phi - \zeta$ (the cylindrical
angle offset), of the Jacobian (`gmnc`, `gmns`), and the profiles $\iota$,
$G$ (`bvco`) and $I$ (`buco`). Every harmonic is
$c\cos(m\theta - n\zeta) + s\sin(m\theta - n\zeta)$ with $n$ in units of the
full torus (booz_xform convention, $n$ a multiple of $N_{fp}$).

{func}`fax.surface_grid` samples these on a uniform $(\theta,\zeta)$ grid
covering the whole torus. Angle derivatives are exact (term by term);
radial derivatives of the harmonics use second-order central differences in
$s$, one-sided at the first and last surface. The position
$\mathbf r = (R\cos\phi, R\sin\phi, Z)$ with $\phi = \zeta + p$ gives the
covariant basis $\mathbf r_i = \partial_i\mathbf r$ for $u^i = (s,\theta,\zeta)$,

$$
g_{ij} = \mathbf r_i\cdot\mathbf r_j,\qquad
g^{ij} = \nabla u^i\cdot\nabla u^j = (g_{ij})^{-1},\qquad
\sqrt g = \mathbf r_s\cdot\mathbf r_\theta\times\mathbf r_\zeta ,
$$

and $|\nabla s|^2 = g^{ss}$. The grid also returns
$\partial_s|B|$, $\partial_\theta|B|$ and $\partial_\zeta|B|$. Two Jacobians
appear below: the continuum uses the Boozer one stored in the file
($\sqrt g_B$, `SurfaceGrid.jacobian`), the global operator the embedding one
($\sqrt g$, `SurfaceGrid.sqrt_g`); they agree up to the truncation of the
Boozer spectrum.

**Radial labels.** `boozmn` files store half-mesh rows `j = jlist - 1`, so
{func}`fax.read_boozmn` sets $s = (j_\mathrm{list} - 1.5)/(n_s - 1)$.
{meth}`fax.Geometry.full_mesh` averages neighbouring half-mesh surfaces onto
the full mesh, as STELLGAP and AE3D do.

**Analytic torus.** {func}`fax.circular_torus` is a concentric circular
tokamak to first order in $\varepsilon = \epsilon_B\sqrt s$:
$|B| = B_0(1 - \varepsilon\cos\theta)$, $G = B_0R_0$, $I = 0$,
$\sqrt g_B = G/B^2 \approx (G/B_0^2)(1 + 2\varepsilon\cos\theta)$, and circles
of radius $r = a\sqrt s$ in the geometric angle $\theta + \varepsilon\sin\theta$
(the $m = 2$ terms of $R$ and $Z$). With $\epsilon_B = a/R_0$ this is the ad hoc
equilibrium of Könies et al. (2018); with $\epsilon_B = 0$ the field is uniform.

## 2. The shear-Alfvén continuum

At low $\beta$, for incompressible shear-Alfvén perturbations, the radial
displacement $E$ of a continuum mode on one surface obeys (Spong, Sanchez &
Weller 2003, eq. 5; Paul et al. 2025, eq. 2.3)

$$
\mu_0\rho\,\omega^2\frac{|\nabla s|^2}{B^2}E
+ \mathbf B\cdot\nabla\!\left(\frac{|\nabla s|^2}{B^2}\,\mathbf B\cdot\nabla E\right) = 0 .
$$ (continuum-ode)

Write $E = \sum_j x_j e_j$ with $e_j = e^{i(m_j\theta - n_j\zeta)}$. By
{eq}`bdotgrad`, $\mathbf B\cdot\nabla e_j = i\,k_{\parallel j} B\, e_j$ with

$$
k_{\parallel j} = \frac{(n_j - m_j\iota)\,|B|}{G + \iota I}\quad(\text{up to the sign, which drops out}).
$$

Multiply {eq}`continuum-ode` by $\sqrt g_B\,e_i^*$, average over the surface,
$\langle a\rangle = (4\pi^2)^{-1}\oint\oint a\,d\theta\,d\zeta$, and integrate
the second term by parts (the surface operator $\sqrt g_B\,\mathbf B\cdot\nabla$
is anti-self-adjoint, since $\nabla\cdot\mathbf B = 0$). This gives the
generalized Hermitian eigenproblem $K x = \omega^2 M x$ with

$$
K_{ij} = \left\langle \frac{|\sqrt g_B|\,|\nabla s|^2}{\mu_0}\,
         k_{\parallel i}k_{\parallel j}\,e_i^*e_j\right\rangle ,\qquad
M_{ij} = \left\langle \frac{\rho\,|\sqrt g_B|\,|\nabla s|^2}{B^2}\,e_i^*e_j\right\rangle .
$$ (continuum-galerkin)

This is the STELLGAP weighting: the metric factor $|\nabla s|^2/B^2$ and the
Jacobian enter both matrices, and $K$ carries $B^2 k_\parallel^2/\mu_0$
(Spong et al. 2003). {func}`fax.continuum_matrices` evaluates the averages on
the uniform grid (default 32 × 32), which is exact for band-limited
integrands (a test checks 32 × 32 against 64 × 64 to 1e-9), and symmetrizes
both matrices. {func}`fax.continuum` solves each surface with a Cholesky
reduction, $L = \mathrm{chol}(M)$, $L^{-1}KL^{-\dagger}y = \omega^2y$, and
returns $f = \sqrt{\max(\omega^2,0)}/2\pi$ sorted on each surface.

**Uniform field.** For $\epsilon_B = 0$ the matrices are diagonal and
$\omega = |n - m\iota|\,v_A/R_0$, $v_A = B_0/\sqrt{\mu_0\rho}$, exactly.

**Mode families.** In a stellarator with $N_{fp}$ periods, $n$ couples only to
$n + kN_{fp}$, so a family is a list `n = [n0, n0 + Nfp, n0 - Nfp, ...]` in
{func}`fax.mode_family`. Paul et al. (2025) normalize by
$\omega_A = \langle B^2\rangle/((G + \iota I)\sqrt{\mu_0\rho})$.

## 3. Gap formulas

Two cylindrical branches $(m, n)$ and $(m + \Delta m, n + \Delta n N_{fp})$,
$\omega = |n - m\iota|v_A/R$, cross where
$n - m\iota = -(n + \Delta n N_{fp} - (m + \Delta m)\iota)$, i.e.

$$
\iota = \frac{2n + \Delta n\,N_{fp}}{2m + \Delta m},\qquad
\frac{\omega R}{v_A} = \frac{|\Delta m\,n - \Delta n\,m\,N_{fp}|}{2m + \Delta m}.
$$ (gap-location)

The Fourier component of the coupling coefficient with the helicity
$(\Delta m, \Delta n N_{fp})$ splits the crossing into a gap:

| Gap | $(\Delta m, \Delta n)$ | coupling | example |
|---|---|---|---|
| TAE | $(1, 0)$ | toroidicity $\cos\theta$ | $\iota = 2n/(2m+1)$, $\omega R/v_A = n/(2m+1)$ |
| EAE | $(2, 0)$ | ellipticity $\cos2\theta$ | $\iota = n/(m+1)$ |
| MAE | $(0, \neq 0)$ | mirror $\cos N_{fp}\zeta$ | $\iota = (2n + \Delta n N_{fp})/2m$ |
| HAE | both $\neq 0$ | helical $\cos(\theta - N_{fp}\zeta)$ | |

To first order the gap is centred on the crossing frequency with full width
$2\omega_0|\epsilon_{\Delta m,\Delta n}|$, where $\epsilon$ is the
complex Fourier coefficient of the relative coupling coefficient
(half its cosine amplitude) (Paul et al. 2025,
eqs. 4.5 and 4.8; Kolesnichenko et al. 2001, eq. 38). {func}`fax.gap_location`
returns $(\iota, \omega R/v_A)$ of {eq}`gap-location`; {func}`fax.gaps` returns
the centre and width of the widest gap between adjacent sorted branches on
each surface (any family, any gap type).

## 4. Global eigenmodes: the AE3D weak form

AE3D (Spong, D'Azevedo & Todo 2010, eqs. 3–7) writes the reduced ideal-MHD
shear-Alfvén equation for the electrostatic potential $\phi$ in variational
form. With $u = \mathbf B\cdot\nabla\phi/B$ and a test function $\tilde\phi$,

$$
\omega^2\int \frac{\mu_0\rho}{B^2}\,\nabla\tilde\phi^*\cdot\nabla\phi\,dV
= \int \nabla\tilde u^*\cdot\nabla u\,dV
\;[\,+\,K + P\,].
$$ (weak-form)

The left side is the inertia (kinetic energy of the $E\times B$ flow), the
right side the field-line bending energy. The integrals use the full
contravariant metric, so both sides are coordinate-free. The bracket is
optional (`finite_beta=True`, off by default).

### Finite-$\beta$ terms

With $\boldsymbol\xi_\perp = \mathbf b\times\nabla\phi/(-i\omega B)$ and
$\mathbf Q_\perp = \nabla A_\parallel\times\mathbf b$, $A_\parallel = u/i\omega$,
the $J_\parallel$ and pressure terms of the ideal-MHD energy,
$-J_\parallel(\boldsymbol\xi_\perp^*\times\mathbf b)\cdot\mathbf Q_\perp$ and
$-2(\boldsymbol\xi_\perp\cdot\nabla p)(\boldsymbol\kappa\cdot\boldsymbol\xi_\perp^*)$
(Freidberg), become, in the units of {eq}`weak-form`,

$$
K = \int\sigma\,\mathbf b\cdot(\nabla\tilde\phi^*\times\nabla u)\,dV,\qquad
P = -2\mu_0\int p'\,\frac{(\mathbf b\times\nabla\phi\cdot\nabla s)\,
(\mathbf b\times\nabla\tilde\phi^*\cdot\boldsymbol\kappa)}{B^2}\,dV,
$$

with $\sigma = \mu_0\mathbf J\cdot\mathbf B/B^2$ and
$\boldsymbol\kappa = \nabla_\perp(\mu_0p + B^2/2)/B^2$. Integrating $K$ by
parts gives AE3D's third term, $-\int u\,\mathbf b\cdot(\nabla\tilde\phi^*\times\nabla\sigma)\,dV$
(the same term follows from the vorticity equation); the form above needs no
derivative of $\sigma$. FAX assembles the Hermitian part of $K$ and $P$, as
AE3D version 30 does with its own symmetrized terms; it does not copy
AE3D's matrix elements. With $\mathbf B = B_s\nabla s + I\nabla\theta + G\nabla\zeta$,
$\mathbf b\cdot(\nabla a\times\nabla c) = \mathbf B_{\rm cov}\cdot(\partial a\times\partial c)/(B\sqrt g)$
on covariant components, so both terms are tensor contractions like the
bending term.

The equilibrium quantities come from $G$, $I$ and $|B|$ alone. The radial
force balance in Boozer coordinates,

$$
(\iota\partial_\theta + \partial_\zeta)B_s = G' + \iota I' + \mu_0p'\,\frac{G + \iota I}{B^2},
$$

fixes $\mu_0p'$ by solvability (its surface average) and $B_s$ (the
Pfirsch–Schlüter part) by dividing by $i(\iota m + n)$ in Fourier space,
skipping resonant harmonics. Then
$\sigma = [I(\partial_\zeta B_s - G') - G(\partial_\theta B_s - I')]/(\sqrt g B^2)$.
Both vanish for a vacuum field ($G$, $I$ constant). {func}`fax.circular_torus`
has $I = 0$ unless `current=True` ($I = \iota s a^2B_0/R_0$) or a `pressure`
polynomial is given, which also lowers $G$ so that
$G' + \iota I' = -\mu_0p'R_0/B_0$.

### Fourier–finite-element basis

$$
\phi(s,\theta,\zeta) = \sum_{k=1}^{N}\sum_{j=1}^{H} y_{kj}\,f_k(s)\,e_j(\theta,\zeta),
$$

where $f_k$ are linear hat functions on the nodes $0 < s_1 < \dots < s_N < 1$
(the geometry surfaces, with the axis and the edge added as end nodes). Then

$$
\nabla(f e_j) = f'\,\mathbf X_{0j} + f\,\mathbf X_{1j},\qquad
\mathbf X_{0j} = e_j\nabla s,\quad
\mathbf X_{1j} = i e_j\left(m_j\nabla\theta - n_j\nabla\zeta\right).
$$

For the bending term, {eq}`bdotgrad` gives $u_j = c_j f e_j$ with

$$
c_j = \frac{i(\iota m_j - n_j)\,|B|}{G + \iota I},
$$

and therefore $\nabla(c_jfe_j) = f'\,\mathbf W_{0j} + f\,\mathbf W_{1j}$,

$$
\mathbf W_{0j} = c_je_j\nabla s,\qquad
\mathbf W_{1j} = e_j\left[\partial_sc_j\,\nabla s
 + (\partial_\theta c_j + im_jc_j)\nabla\theta
 + (\partial_\zeta c_j - in_jc_j)\nabla\zeta\right],
$$

$$
\partial_sc_j = \frac{i}{G + \iota I}\Big[\iota'\,m_j|B|
 + (\iota m_j - n_j)\Big(\partial_s|B| - |B|\frac{(G + \iota I)'}{G + \iota I}\Big)\Big],\qquad
\partial_{\theta,\zeta}c_j = \frac{i(\iota m_j - n_j)\,\partial_{\theta,\zeta}|B|}{G + \iota I}.
$$ (shear-term)

The $\iota' m_j$ term is the magnetic shear. It is physical: $u$ depends on
$s$ through $\iota(s)$, and a real-space quadrature of $\int|\nabla u|^2dV$
built from $|B|$, $\iota$ and $g_{ij}$ alone agrees with $F$ to 0.3% with this
term and is off by 9–20% without it (see {doc}`validation`). Radial
derivatives of $\iota$ and $G + \iota I$ use the same finite differences as
the harmonics.

### Coefficient blocks and element matrices

For a weight $w$ and vectors $\mathbf V_0, \mathbf V_1$ (either
$\mathbf X$ with $w = \mu_0\rho|\sqrt g|/B^2$, or $\mathbf W$ with
$w = |\sqrt g|$), each surface gives three $H\times H$ blocks

$$
A_{ij} = \langle w\,\mathbf V_{0i}^*\cdot\mathbf V_{0j}\rangle,\quad
B_{ij} = \langle w\,\mathbf V_{0i}^*\cdot\mathbf V_{1j}\rangle,\quad
D_{ij} = \langle w\,\mathbf V_{1i}^*\cdot\mathbf V_{1j}\rangle ,
$$

with $\mathbf a^*\cdot\mathbf b = a_i^*g^{ij}b_j$ on the
$(\nabla s,\nabla\theta,\nabla\zeta)$ components. As in AE3D, each block is
averaged over a radial element (the mean of its two end surfaces; the axis and
edge elements reuse the first and last surface). On an element of width $h$
with left and right hat functions ($f_L' = -1/h$, $f_R' = 1/h$,
$\int f_L^2 = \int f_R^2 = h/3$, $\int f_Lf_R = h/6$,
$\int f_L'f_L = -\tfrac12$, ...), exact integration gives

$$
\begin{aligned}
E_{LL} &= \frac{A}{h} - \frac{B + B^\dagger}{2} + \frac{Dh}{3}, &
E_{RR} &= \frac{A}{h} + \frac{B + B^\dagger}{2} + \frac{Dh}{3},\\
E_{LR} &= -\frac{A}{h} - \frac{B - B^\dagger}{2} + \frac{Dh}{6}, &
E_{RL} &= -\frac{A}{h} + \frac{B - B^\dagger}{2} + \frac{Dh}{6}.
\end{aligned}
$$

Summing elements gives the Galerkin problem

$$
F y = \omega^2 G y ,
$$ (global-problem)

with $F$ (bending) and $G$ (inertia) Hermitian and block tridiagonal: $N$
diagonal blocks $E_{RR}^{(k)} + E_{LL}^{(k+1)}$ and $N - 1$ off-diagonal
blocks $E_{LR}$ ({func}`fax.global_matrices`, which returns the two
`Blocks`). The angle averages carry no $4\pi^2$, which cancels in
{eq}`global-problem`.

### Boundary conditions

- **Edge:** $\phi = 0$ at $s = 1$ (the edge node is dropped).
- **Axis:** regularity $\phi_m \sim s^{|m|/2}$ is imposed in AE3D's form:
  harmonics with $m \neq 0$ vanish on the axis (the axis node is dropped),
  and $m = 0$ harmonics are flat between the axis and the first surface,
  $\phi(0) = \phi(s_1)$. With $P = \mathrm{diag}(m_j = 0)$ this adds
  $E_{RL}P + PE_{LR} + PE_{LL}P$ of the axis element to the first diagonal
  block.

The singular $f'f'$ blocks $A/h$ reproduce the local continuum of
{eq}`continuum-galerkin`: where $\omega$ meets a continuum branch the
solution develops a jump in $\phi'$ localized to one cell, so the discrete
spectrum fills the continuum with modes of width $\propto 1/N$, and true gap
modes appear as isolated eigenvalues.

## 5. Block-tridiagonal shift-invert solver

The modes of interest lie in the interior of the spectrum, near a target
$\sigma = (2\pi f_\mathrm{target})^2$. {func}`fax.eigenmodes` with
`target_hz` uses shift-invert subspace iteration
(`fax.eigen.shift_invert`):

1. Factor $A = F - \sigma G$ by block LU (the Thomas algorithm with
   $H\times H$ blocks, a `lax.scan` forward and backward), at cost
   $O(NH^3)$ instead of $O((NH)^3)$:
   $S_1 = A_{11}$, $S_k = A_{kk} - A_{k,k-1}S_{k-1}^{-1}A_{k-1,k}$,
   $z_k = b_k - A_{k,k-1}S_{k-1}^{-1}z_{k-1}$, then
   $x_k = S_k^{-1}(z_k - A_{k,k+1}x_{k+1})$ backwards.
2. Iterate a block of $p = \min(2\,\mathrm{count} + 8, NH)$ vectors (seeded
   with a fixed random block) `iters = 40` times:
   $Y \leftarrow \mathrm{qr}\big((F - \sigma G)^{-1} G\,Y\big)$. Eigenvalues
   near $\sigma$ have the largest $|\lambda - \sigma|^{-1}$ and dominate.
3. Rayleigh–Ritz: solve $(Y^\dagger FY)c = \lambda(Y^\dagger GY)c$, keep the
   `count` values nearest $\sigma$ in ascending order, and return
   $G$-normalized eigenvectors $y = Yc$.

Without a target, `eigenmodes` assembles the dense matrices and solves all
$NH$ modes. A test checks both solvers agree to 1e-10. The CTOK case
($N = 99$, $H = 13$) assembles in 0.018 s and finds 6 modes in 0.067 s once
compiled; the dense solve takes 1.3 s.

## 6. Eigenvalue derivatives (Hellmann–Feynman)

For $Fy = \lambda Gy$ with Hermitian $F$, positive-definite $G$ and
$y^\dagger Gy = 1$, differentiate and multiply by $y^\dagger$; the terms in
$\delta y$ cancel by symmetry, leaving

$$
\delta\lambda = y^\dagger\left(\delta F - \lambda\,\delta G\right)y .
$$ (hellmann-feynman)

FAX implements this as a `custom_jvp`, so no derivative passes through the
iteration or the dense `eigh`:

- `fax.eigen.targeted_eigenvalues` (used by {func}`fax.frequency_hz` with a
  target) applies {eq}`hellmann-feynman` to the shift-invert pairs;
- {func}`fax.eigenvalue_mean` applies it to dense eigenvalues
  `start <= i < stop` and returns the **mean**,
  $\bar\lambda = \frac1k\sum_{i}\lambda_i$, whose derivative
  $\frac1k\sum_i y_i^\dagger(\delta F - \lambda_i\delta G)y_i$ is the trace of
  the perturbation projected on the cluster. It stays smooth when
  eigenvalues inside the cluster cross or are degenerate (where single
  eigenvalues have kinks and undefined eigenvectors), as long as the cluster
  is separated from the rest of the spectrum.

{func}`fax.frequency_hz` returns $\sqrt{\bar\lambda}/2\pi$: the frequency of
the mean $\omega^2$ of the selected eigenvalues (for one eigenvalue, its
frequency). Derivatives of $F$ and $G$ themselves (with respect to geometry,
$\iota$, density) are ordinary JAX derivatives of the assembly.

(vmex-gradients)=
## 7. vmex boundary gradients

{func}`fax.prepare_design` and {func}`fax.design_geometry` chain

$$
\text{boundary offsets } p \;\to\; x^*(p) \;\to\; \text{Boozer harmonics} \;\to\; (F, G) \;\to\; \omega^2 .
$$

- **Equilibrium.** vmex's `implicit.solve_implicit` returns the converged
  VMEC state $x^*$ with $R(x^*, p) = 0$ (the discrete force balance). Its
  derivative uses the implicit-function theorem,
  $\partial x^*/\partial p = -(\partial_xR)^{-1}\partial_pR$, applied in reverse
  mode as one adjoint solve per scalar objective, so the gradient is that of
  the converged equilibrium, not of a truncated iteration (Antonsen, Paul &
  Landreman 2019 for the shape-gradient idea).
- **Parameters.** Names `rc|rs|zc|zs` + `m` + `n` select `RBC`, `RBS`, `ZBC`,
  `ZBS` entries; `"zs01"` is `ZBS(n=1, m=0)`. Offsets are added to the input's
  coefficients.
- **Boozer transform.** `booz_xform_jax_impl` is traced on the half-mesh
  rows $j \ge 2$, $s = (j - 1/2)/(n_s - 1)$ (row 1 waits for a vmex fix of its
  $R$, $Z$ tables).
- **Radial grid.** {meth}`fax.geometry.Geometry.interpolate` maps that geometry,
  linearly in $s$ and extrapolated to $s = 0, 1$, onto `nodes` uniform FE
  nodes $k/(\text{nodes} + 1)$. The radial domain is always $[0, 1]$, and the
  FE grid refines independently of $n_s$ and `mboz`.
- **Spectrum.** {func}`fax.geometry_frequencies` returns the lowest
  frequencies (dense) or those nearest `target_hz` (shift-invert). Both carry
  the {eq}`hellmann-feynman` JVP, so a smooth objective
  ({func}`fax.window_risk`, {func}`fax.soft_max`) of {func}`fax.design_frequencies`
  is differentiable from the boundary offsets.
- **Gauge.** VMEC's released $m = 1$ constraint combination is an angle gauge.
  The adjoint holds it fixed, while a re-solve lets it drift along its own
  path. Re-solve finite differences therefore differ from the adjoint by the
  response to that drift: 4.4e-4 for $\iota$ wrt `zs01`, about 3e-5 for the
  global frequencies wrt `rc01`, `zs01`, `rc10`, and 2.7e-3 wrt the $m = 1$
  coefficient `zs11`. Adding the drift response closes the $\iota$ gap to 3e-9.

`jax.grad` of any scalar of {func}`fax.design_frequencies` therefore costs one
forward solve plus one adjoint equilibrium solve.

## 8. Perturbative fast-ion drive

The growth rate of a fixed eigenmode is the power the fast ions give to it
over twice its energy (Chen & Zonca 2016, eq. 3.22; NOVA-K and CAS3D-K form),

$$
\gamma_h = -\frac{\operatorname{Im}\delta W_h}{2\omega\,\delta K} = -\frac{P}{2W},\qquad
W = \int\frac{\rho\,|\nabla\phi|^2}{2B^2}\,dV = \frac{2\pi^2}{\mu_0}\,y^\dagger G\,y ,
$$ (gamma-power)

with $G$ the inertia matrix of {func}`fax.global_matrices` (angle-averaged, so
$\int dV = 4\pi^2\int ds\,\langle|\sqrt g|\,\cdot\rangle$) and $y$ the nodal
values of the mode. {func}`fax.growth_rate` evaluates $P$ on the surfaces of a
{class}`fax.Geometry`, for zero orbit width (ZOW), zero Larmor radius and
passing guiding centres, so tokamaks and stellarators share one code path.

**Mode and drift.** $\phi = \sum_h\phi_h(s)e^{i(m_h\theta - n_h\zeta - \omega t)}$,
the convention of {mod}`fax.eigen`. At low $\beta$ the curvature plus
$\nabla B$ drift is
$\mathbf v_d = \frac{m v^2(1 + \xi^2)}{2ZeB^3}\,\mathbf B\times\nabla B$, with
$\xi = v_\parallel/v$. With the Boozer covariant components $B_\theta = I$,
$B_\zeta = G$ ($B_s$ dropped) and the embedding Jacobian $\sqrt g$ of
$(s, \theta, \zeta)$,
$(\mathbf B\times\nabla B)^i = \epsilon^{ijk}B_j\partial_kB/\sqrt g$:

$$
\mathbf v_d\cdot\nabla s \propto \frac{I\partial_\zeta B - G\partial_\theta B}{\sqrt g},\qquad
\mathbf v_d\cdot\nabla\theta \propto \frac{G\,\partial_sB}{\sqrt g},\qquad
\mathbf v_d\cdot\nabla\zeta \propto -\frac{I\,\partial_sB}{\sqrt g},
$$

so the work $Ze\,\mathbf v_d\cdot\delta\mathbf E = -Ze\,\mathbf v_d\cdot\nabla\phi$
of harmonic $h$ is $-m v^2$ times

$$
w_h = \frac{1 + \xi^2}{2B^3\sqrt g}\Big[(I\partial_\zeta B - G\partial_\theta B)\,\phi_h'
  + i\,\partial_sB\,(Gm_h + In_h)\,\phi_h\Big]e^{i(m_h\theta - n_h\zeta)} .
$$

The binormal terms ($\partial_sB$) are the $\cos\theta\,\partial_\theta/r$
part of the large-aspect-ratio vertical drift. Only the overall sign of
$\sqrt g$ is convention, and it cancels in $|w|^2$.

**Orbits and resonances.** A passing guiding centre of pitch
$\lambda = \mu/E \in [0, 1/B_\max)$ follows the field line
$\theta - \iota\zeta = \text{const}$ with
$dt/d\zeta = (G + \iota I)/(v|\xi|B)$, since $B^\zeta = B^2/(G + \iota I)$. Its
mean toroidal transit frequency is
$\omega_\zeta = v/[(G + \iota I)\langle h\rangle]$, $h = 1/(|\xi|B)$ and
$\langle\cdot\rangle$ the Boozer-angle average (the field-line average on an
irrational surface). The angles that advance uniformly in time are
$\vartheta = \theta + \iota\chi$, $\varphi = \zeta + \chi$, with $\chi$ the
periodic solution of the magnetic differential equation

$$
(\iota\partial_\theta + \partial_\zeta)\chi = h/\langle h\rangle - 1,
$$

solved by FFT on the angle grid (a $10^{-2}$ guard on the small divisors
$\iota k_\theta + k_\zeta$). Decomposing the work in $(\vartheta, \varphi)$, the
component $e^{i(j\vartheta - k\varphi)}$ resonates at

$$
\omega = |j\iota - k|\,\omega_\zeta(v, \lambda),\qquad j = m + \ell,\quad k = n + k'N_{fp},
$$ (resonance)

for either sign of $v_\parallel$. The $\ell$ sidebands come from the poloidal
variation of $B$ and of the drift (TAE: $v_\parallel = v_A$, $v_A/3$, ...), the
$k'$ sidebands from the helical harmonics of $B$ (Kolesnichenko et al. 2002;
Heidbrink 2008). `sidebands` (default 3) bounds $|\ell|$ and `helical`
(default 1) bounds $|k'|$; `resolved=True` returns $\gamma$ per $(j, k)$.
The orbit coefficient is

$$
C_{h,jk} = \frac{\langle h\,w_h\,e^{-i(j\vartheta - k\varphi)}\rangle}{\langle h\rangle}.
$$

**Power transfer.** At ZOW, $P_\zeta = m v_\parallel G/B - Ze\psi_p$ with
$\psi_p' = \iota\psi'$ and $\psi' = \langle\sqrt g\,B^2\rangle/(G + \iota I)$
($\psi$ the toroidal flux over $2\pi$). With $E_\parallel = 0$ the radial
$E\times B$ and $v_\parallel\delta B_r$ drifts combine into $-\partial_\zeta\phi$, so
each harmonic carries its own

$$
Q_hF = \frac{\partial F}{\partial E} + \frac{n_h}{\omega Ze\,\iota\psi'}\frac{\partial F}{\partial s}
$$

(for one $n$ this is $\partial_EF - (n/\omega)\partial_{P_\zeta}F$). Linearizing
along unperturbed orbits, the Landau prescription
$\operatorname{Re}[1/(0^+ - i\Delta)] = \pi\delta(\Delta)$, the phase-space
measure $d^3x\,d^3v = d\psi\,d\theta\,d\zeta\,\sum_\sigma\pi v^2dv\,d\lambda\,(G + \iota I)/(|\xi|B)$
and the delta function integrated exactly over speed at each pitch give

$$
\gamma = \frac{\pi^2\mu_0m_f^2}{2\,y^\dagger Gy}\int|\psi'|\,ds\int d\lambda
  \sum_{jk}\frac{v^6\operatorname{Re}\big[\bar C\,\textstyle\sum_hQ_hF\,C_h\big]}{|j\iota - k|\,\hat\omega_\zeta^2}\Big|_{v = v_{jk}},\qquad
\bar C = \sum_hC_h,\quad v_{jk} = \frac{\omega}{|j\iota - k|\,\hat\omega_\zeta},
$$ (gamma-zow)

$\hat\omega_\zeta = \omega_\zeta/v$. Numerics: Gauss–Legendre (`nodes = 32`) in
$\lambda$ up to $1/B_\max$ (the exact sample maximum, so duplicated
toroidal samples change nothing), the
`ntheta` × `nzeta` Boozer grid of {func}`fax.surface_grid`, central
differences in $s$, the trapezoidal rule in $s$ with $\phi = 0$ at the ends.
There is no artificial broadening, so $\gamma$ is smooth in the geometry, the
mode, $\omega$ and the distribution. In an axisymmetric geometry with
`helical = 0` one toroidal grid point (`nzeta = 1`) is exact.

**Reduction to the circular model.** For {func}`fax.circular_torus`
($G = B_0R_0$, $I = 0$, $B = B_0(1 - \varepsilon\cos\theta)$) $w_h$ becomes the
vertical drift $(1 - \lambda B/2)[\sin\theta\,\phi_h' + \cos\theta\,\partial_\theta\phi_h/r]/R$,
$\omega_\zeta/\iota$ the poloidal transit frequency and $Q_hF$ the familiar
$\partial_EF - nq\,\partial_rF/(\omega ZeB_0r)$ (with $n \to -n$ for the mode
convention). The Könies TAE then reproduces the earlier large-aspect-ratio
implementation within 4% ({doc}`validation`).

**Distributions.** {class}`fax.FastIons` is isotropic: a Maxwellian
($F \propto e^{-E/T}$, so $\partial_EF = -F/T$) or a slowing-down
$F \propto 1/(v^3 + v_c^3)$ below the birth speed $v_b$, with the birth edge
smoothed by $\tfrac12\operatorname{erfc}((v - v_b)/0.02v_b)$ so it stays
differentiable ($\partial_EF$ includes the smoothed edge). $F$ is normalized to
$\int F\,d^3v = 1$ with that edge: integrating by parts, the constant is
$3/(4\pi\,\mathbb E[\ln(1 + v^3/v_c^3)])$ with $v \sim \mathcal N(v_b, (0.02v_b)^2/2)$,
done with 24-point Gauss–Hermite.
Any other `kind` raises `ValueError`.
The density may be a profile on $s$; with $\iota\psi' > 0$ an
outward-decreasing density drives $n > 0$, and $\partial_EF < 0$ damps.

**Landau damping.** {func}`fax.ion_landau_damping` is the same integral for
a flat thermal Maxwellian of the bulk ions, with the sign flipped so that a
positive number damps (Fu & Van Dam 1989; Betti & Freidberg 1992). Continuum
damping and electron Landau damping are not computed; subtract them as an
external $\gamma_d$, $\gamma = \gamma_h - \gamma_d$.

**Not included.** Trapped and precessing particles (their resonance is
$\omega = k\langle\omega_d\rangle_b + p\,\omega_b$ with bounce averages; not
yet implemented), finite orbit width, $\delta B_\parallel$ and the covariant
$B_s$ in the drift.

## 9. Smooth objectives

Eigenvalues and gaps are not smooth (crossings, `max`), so the objectives are:

| Function | Definition |
|---|---|
| {func}`fax.window_risk` | $\tfrac12\,\mathrm{mean}_k\,\sigma_k^2$, $\sigma_k = \mathrm{sig}\big(\frac{f_k - f_\mathrm{lo}}{\Delta}\big)\,\mathrm{sig}\big(\frac{f_\mathrm{hi} - f_k}{\Delta}\big)$: a soft count of modes inside a frequency window |
| {func}`fax.soft_max` | $\beta^{-1}\log\sum_k e^{\beta x_k}$ (log-sum-exp), e.g. of growth rates |
| {func}`fax.gap_closure` | $\mathrm{mean}_s\,\mathrm{softplus}(-w_s/\Delta)$ with $w_s$ the widest gap on surface $s$: grows as gaps close |

{func}`fax.drive_risk` is the stability target
$\mathrm{softplus}(\mathrm{LSE}(\gamma_k/\gamma_s) - \gamma_\mathrm{th}/\gamma_s)$ (scale
$\gamma_s$ = `scale`, default 10³ s⁻¹) of the growth rates that
{func}`fax.geometry_growth_rates` returns for the modes nearest `target_hz`
(shift-invert eigenvectors, differentiated through the iteration, then
{func}`fax.growth_rate`). Cluster means
({func}`fax.eigenvalue_mean`, `start:stop` in {func}`fax.frequency_hz`)
keep objectives smooth through crossings inside the cluster.

(continuum-objective)=
## 10. Continuum-gap objective

Gaps open where the flux compression $|\nabla\psi|^2$ varies on a surface
(sections 2–3). Paul et al. (2025, eq. 4.28) write the coupling as

$$
\epsilon = \frac{|\nabla s|^2}{\langle|\nabla s|^2\rangle} - 1
         = \sum_{\delta m,\,n} \epsilon_{\delta m,n}\, e^{i(\delta m\,\theta - n\zeta)},
$$

in Boozer angles, with $\langle\cdot\rangle$ the angle average and $n$ the full
toroidal number ($N_P\,\delta n$ in the paper). {func}`fax.compression_harmonics`
takes $|\nabla s|^2$ from {func}`fax.surface_grid` and returns $\epsilon_{\delta m,n}$
from one 2-D FFT per surface. The harmonic $(\delta m, n)$ opens a gap centred at
$\omega^{(0)}/\omega_A = |\iota\,\delta m - n|/2$ of full (edge to edge) width
$2\omega^{(0)}|\epsilon_{\delta m,n}|$ (eqs. 4.5, 4.16; the conjugate $(-\delta m, -n)$ makes the real amplitude $2|\epsilon|$). Gaps above the
quasisymmetry-helicity gap are the ones passing fast ions near birth energy can
resonate with, so Paul's eq. 7.4 penalizes the squared full widths $(2\omega^{(0)}|\epsilon|/\omega_A)^2 = |\epsilon|^2(\iota\delta m - n)^2$ of those:

$$
f_\mathrm{cont} = \sum_s \sum_{|\iota\delta m - n| > |\iota M - N N_P|}
                  |\epsilon_{\delta m,n}|^2\,(\iota\,\delta m - n)^2,
$$

{func}`fax.continuum_cost`, with helicity $(M, N)$: $(1, 0)$ for QA and
$(1, \pm1)$ for QH (the sign follows the Boozer $|B|$ spectrum: LP-QH with
$\iota < 0$ has $|B| \sim \cos(\theta + 4\zeta)$, i.e. $(1, -1)$). Each term is
the square of a full gap width, which grows with the gap frequency, so wide
high-frequency gaps dominate. The sum runs over both $\pm(\delta m, n)$; the cut is a hard mask
(smooth almost everywhere: a term only enters or leaves when $\iota$ crosses
a threshold). The full objective of Paul's eq. 7.1,

$$
f = (A - A^*)^2 + f_{QS} + f_\iota + f_\mathrm{cont},\qquad
f_\iota = \sum_s \min(|\iota| - \iota_\mathrm{min}, 0)^2,
$$

takes $A$ and $f_{QS}$ from vmex (`implicit.aspect_ratio`,
`QuasisymmetryRatioResidual.total_state`) on the same implicit state:
{func}`fax.design_state` returns `(state, runtime)` and
{func}`fax.state_geometry` the Boozer geometry, so one adjoint covers every
term. For the global spectrum, {func}`fax.window_risk` of
{func}`fax.geometry_frequencies` moves targeted modes out of a frequency window
instead (section 9).
