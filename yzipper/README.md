# Y-zipper — 3-strip flexible↔rigid element (paper-faithful rebuild)

*Rebuilt 2026-07-23 to follow the paper's ACTUAL algorithm (an earlier version used a
made-up "differential pitch → radius" rule + rectangular teeth, which did not match the
paper — the shapes came out wrong). Now grounded in Jiaji Li et al., "Y-zipper", ACM
CHI '26 — [doi:10.1145/3772318.3790723](https://dl.acm.org/doi/10.1145/3772318.3790723),
§3.1 (tooth geometry) + Appendix A.3 (primitive formulas).*

*A stiffness element, not a linear actuator — the actuator sibling is the 2-strip zip
chain in [`../zipchain/`](../zipchain/). The "actuator" here is the slider.*

> **Hardware in progress (2026-07-22):** test-printing a community "extended / optimized"
> three-sided Y-zipper ([MakerWorld 2888456](https://makerworld.com/en/models/2888456-y-zipper-extended-optimized-three-sided-zipper)).
> The measured per-tooth engagement force and the screw constant `k` are the two
> calibration-pending inputs the print pins down.

---

## Tooth geometry (§3.1) — what was wrong before

Each vertical **segment** is three teeth (a, b, c) with a **wave-like profile** that
**cyclically overlap** — left of a over right of b, b over c, c over a — a mutually
supporting ring that turns three flexible strips into a rigid triangular prism. Each
tooth carries a **ball node (∅1 = 2.4 mm)** on top and a **socket (∅2 = 3.0 mm)** below
(tol 0.3 mm) for shear/alignment, and **compliant bridges** (thickness D; TPU 0.6–2.4,
PLA 0.4–1.2) join segments and carry tension when zipped. *(The old model used uniform
rectangular castellations with none of this — hence "the output doesn't look right".)*

## Programming shapes = varying the teeth (Appendix A.3)

The four one-DOF primitives are **not** made by changing pitch; they come from the tooth
geometry:

| Primitive | Mechanism | Formula |
|---|---|---|
| **Straight** | uniform teeth | `z = v·t` |
| **Bend** | teeth at interface α–β **thinner** than α–γ; accumulate `T1, T2` | **θ = 2(T1−T2)/(√3·w)**,  **R = (T1+T2)/(2θ)** |
| **Screw** | shift each tooth's lower side up by Δt → ball-normal rotates | **θ_z = k·Δt**  (−1.5t ≤ Δt ≤ 2.5t),  L_z = √(L²−(r·θ_z)²), r = w/√3 |
| **Coil** | bend **+** axial rise Δz per segment | helix R, pitch h from `T1,T2,T3,Δz` |
| **Series** | primitives compose via homogeneous transforms | `r(t) = Σ Tᵢ₋₁·rᵢ(tᵢ)` |

The bend is a *bimetallic-strip* effect: a difference in accumulated tooth **thickness**
across the two interfaces, over the interface separation √3·w/2 (the triangle height),
bends the axis. `yzipper.py` also exposes the **design inverse** the Grasshopper tool
gives users: the (T1−T2) needed for a target angle (bend 90° → T1−T2 = 34.0 mm at w=25).

## Stiffness — the paper's MEASURED values (not an idealized calc)

Three-point bending (§9.2): **EI_strip ≈ 1.9×10³ → EI_rod ≈ 3.1×10⁵ N·mm² ≈ 160×**.
Beam stiffness `k = 48·EI/L³`; max load rises with bridge D (11 kg @ 0.8 mm → 18 kg @
2.0 mm). *(An idealized closed-thin-wall calc overpredicts ~400× because teeth + bridges
are not a solid tube — so the model is anchored to the measured numbers.)*

## Files

- **`yzipper.py`** — the primitive algorithm: `inc_straight/inc_bend/inc_screw/inc_coil`
  build each primitive from the Appendix A.3 equations, `_integrate` composes them via
  homogeneous transforms, and `out/primitives.png` **reproduces the paper's Fig. 6**
  (straight prism, bend arch, coil helix, screw twist, and a straight+90°-bend series).
  Calibrated stiffness/load report.
- **`cad.py`** — corrected tooth geometry: rounded **wave teeth + ball nodes + sockets +
  bridges** (`strip_straight`), and a **curved arc flat-pattern with differential tooth
  spacing** (`strip_bend`) that embodies the θ = 2(T1−T2)/(√3·w) bend; plus the 3-way
  `slider` and a `segment` prism. All watertight solids.
- **`sim.py`** — MuJoCo viewer: the closed rod is instanced along a paper-equation
  centreline; the slider zips base→tip and the **rigid (zipped) front advances** while
  the unzipped part stays translucent. `make sim-yzipper` (bend) / `sim.py coil`.

`make yzipper / yzipper-cad / sim-yzipper`.

---

## Sources
- Paper (open access PDF): https://groups.csail.mit.edu/hcie/files/research-projects/y-zipper/y-zipper.pdf
- Paper (ACM CHI '26): https://dl.acm.org/doi/10.1145/3772318.3790723
- Project page / video: https://hcie.csail.mit.edu/  · https://www.youtube.com/watch?v=AWig98GVIno
- Printable model being tested: https://makerworld.com/en/models/2888456-y-zipper-extended-optimized-three-sided-zipper
