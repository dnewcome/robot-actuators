"""
Layer-B model of a ZIP-CHAIN LINEAR ACTUATOR (2-strip / Tsubaki-style rigid chain).

Two chain strands are fed into a sprocket head where they interlock like a zipper
into a single rigid column; run the sprocket backward and they separate and re-coil.
Unlike a cable (pull only) the meshed column carries COMPRESSION, so it pushes and
pulls — and the mesh self-locks, so it holds a load with the motor unpowered.

This is the linear counterpart that trades the *fluidic* gearing of
`linear/actuator.py` for a *geometric* one: a compact pair of coils deploys into a
rigid strut. The whole design trade lives in one place —

    THE BINDING CONSTRAINT IS COLUMN BUCKLING, NOT THE MOTOR.

The deployed free length between the sprocket head and the load is an Euler strut, so
its push capacity FALLS as it extends:  F_cr(L) = π²·E·I_col / (k·L)².  Below a knee
length L* the actuator is drive-/mesh-limited (a flat ceiling); beyond L* it is
buckling-limited (falls as 1/L²).  The push–stroke envelope is therefore ASYMMETRIC
from the pull side (which is tension-limited by the strand strength) — this module's
job is to expose that envelope honestly, the way the cycloidal validator exposes
radial real-estate before any geometry is meshed.

Three design levers, all `Param`s:
  1. COLUMN STIFFNESS  E·I_col  — section (w×h) × an engagement fraction I_engage < 1
     (meshed teeth are not a solid bar).  Sets where buckling starts to bite.
  2. SPROCKET PITCH RADIUS  r_p  — the "gear ratio": trades thrust for deploy speed
     (F = η·τ/r_p, v = ω·r_p).  Power is conserved across it, like any gear.
  3. END FIXITY  k  — how the load end is supported (k=2 fixed-free cantilever is the
     conservative default; a guided load end approaches k=1 and quadruples F_cr).

Everything is analytical first-order (Euler buckling, rack-and-pinion drive, lumped
strengths) — good for the F–stroke envelope, the buckling knee, deploy speed, and the
compactness/hold story; NOT a contact/mesh-FEA model.  Absolute strengths
(link_break, mesh_crush) and the engagement fraction are the calibration-pending
inputs, flagged as such.

    ../.venv/bin/python zipchain/zipchain.py        # report + out/envelope.png
"""

from dataclasses import dataclass
from math import pi, sin, sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


@dataclass
class ZipChainParams:
    # --- Chain / meshed-column geometry ----------------------------------------
    pitch_mm: float = 10.0           # chain pitch (one link) [mm]
    strands: int = 2                 # a zip chain is two strands
    col_w_mm: float = 16.0           # meshed-column width, seam/strong axis [mm]
    col_h_mm: float = 12.0           # meshed-column height, weak (buckling) axis [mm]
    I_engage: float = 0.35           # fraction of a SOLID bar's I the meshed teeth realize (0..1)
    link_min_bend_r_mm: float = 22.0 # tightest coil radius ONE strand can articulate to [mm]

    # --- Sprocket / drive head -------------------------------------------------
    sprocket_teeth: int = 12         # teeth on the zip sprocket
    drive_eff: float = 0.90          # sprocket-mesh efficiency (~0.9 with dedicated sprockets)

    # --- Column material (printed, effective) ----------------------------------
    E_col_MPa: float = 2000.0        # EFFECTIVE bending modulus of the meshed column
                                     # (PLA ~3.5 GPa solid, knocked down for the mesh/print)
    end_fixity_k: float = 2.0        # Euler effective-length factor
                                     # (2 = fixed-free cantilever [conservative]; 1 = guided/pinned)

    # --- Strength limits -------------------------------------------------------
    link_break_N: float = 1500.0     # tensile strength of a strand — sets the PULL cap
    mesh_crush_N: float = 3000.0     # compressive crush of the meshed teeth — push cap before buckling

    # --- Stroke / drive input --------------------------------------------------
    stroke_mm: float = 300.0         # deployed usable stroke [mm]
    retract_free_mm: float = 20.0    # minimum free length at full retract (head standoff) [mm]
    motor_torque_Nm: float = 2.0     # torque AT THE SPROCKET (after any reduction) [N·m]
    motor_rpm: float = 300.0         # sprocket speed at that torque [rpm]

    # ---- derived geometry -----------------------------------------------------
    @property
    def r_pitch_m(self) -> float:
        """Sprocket pitch radius r_p = p / (2 sin(pi/z)) [m]."""
        return (self.pitch_mm / (2.0 * sin(pi / self.sprocket_teeth))) / 1000.0

    @property
    def I_col_m4(self) -> float:
        """Weak-axis area moment of the meshed column [m^4], with engagement knock-down.

        Weak axis bends about the wide dimension → I_weak = w·h^3/12 (h is the small side).
        """
        w = self.col_w_mm / 1000.0
        h = self.col_h_mm / 1000.0
        I_solid = min(w * h**3, h * w**3) / 12.0   # weak axis = smaller of the two
        return self.I_engage * I_solid

    @property
    def EI(self) -> float:
        """Column bending stiffness E·I [N·m^2]."""
        return (self.E_col_MPa * 1e6) * self.I_col_m4

    # ---- kinematics -----------------------------------------------------------
    @property
    def deploy_speed_mms(self) -> float:
        """Linear deploy/thrust speed v = omega·r_p [mm/s]."""
        omega = self.motor_rpm * 2.0 * pi / 60.0
        return omega * self.r_pitch_m * 1000.0

    # ---- forces ---------------------------------------------------------------
    @property
    def F_drive_N(self) -> float:
        """Thrust the sprocket can source, rack-and-pinion style: F = eff·tau/r_p [N]."""
        return self.drive_eff * self.motor_torque_Nm / self.r_pitch_m

    def F_buckle_N(self, free_len_mm: float) -> float:
        """Euler critical thrust of the deployed column at this free length [N]."""
        L = max(free_len_mm, 1e-3) / 1000.0
        return (pi**2) * self.EI / (self.end_fixity_k * L) ** 2

    def push_capacity_N(self, free_len_mm: float) -> float:
        """Usable PUSH force = min(drive, mesh crush, buckling) at this extension [N]."""
        return min(self.F_drive_N, self.mesh_crush_N, self.F_buckle_N(free_len_mm))

    def pull_capacity_N(self, free_len_mm: float) -> float:
        """Usable PULL force = min(drive, strand strength) — tension-limited, ~flat in L [N]."""
        return min(self.F_drive_N, self.link_break_N)

    @property
    def buckle_knee_mm(self) -> float:
        """Free length L* where buckling first drops below the drive/mesh ceiling [mm].

        Beyond L* the push envelope is buckling-limited (falls as 1/L^2).
        """
        ceil = min(self.F_drive_N, self.mesh_crush_N)
        # F_cr(L*) = ceil  ->  L* = pi·sqrt(EI/ceil)/k
        return (pi * sqrt(self.EI / ceil) / self.end_fixity_k) * 1000.0

    def hold_capacity_N(self, free_len_mm: float) -> float:
        """Static load held with the MOTOR OFF (mesh self-locks) — compression side [N].

        Self-locking removes the motor from the equation; what's left is the column's
        own buckling limit (and the mesh crush cap).
        """
        return min(self.mesh_crush_N, self.F_buckle_N(free_len_mm))

    # ---- packaging ------------------------------------------------------------
    @property
    def store_footprint_mm(self) -> float:
        """Rough stored diameter: two coils each holding ~half the deployed chain [mm]."""
        # chain length per strand ~ stroke; coiled at min bend radius, area-equivalent OD
        chain_len = self.stroke_mm
        r = self.link_min_bend_r_mm
        # turns to store chain_len at mean radius r: crude spiral -> outer radius growth
        # area of spiral annulus ~ chain_len * pitch_thickness; approximate OD:
        thick = self.col_h_mm
        r_out = sqrt(r**2 + (chain_len * thick) / pi)
        return 2.0 * r_out

    @property
    def compactness_vs_rigid(self) -> float:
        """Stored length / stroke — a rigid ballscrew of equal stroke is >= 1·stroke long."""
        return self.store_footprint_mm / self.stroke_mm

    # ---- feasibility ----------------------------------------------------------
    def checks(self) -> list[tuple[str, bool, str]]:
        out = []
        push_full = self.push_capacity_N(self.stroke_mm + self.retract_free_mm)
        out.append((
            "push usable at full extension",
            push_full > 0.15 * self.F_drive_N,
            f"{push_full:.0f} N at L={self.stroke_mm + self.retract_free_mm:.0f} mm "
            f"(drive can source {self.F_drive_N:.0f} N)",
        ))
        out.append((
            "sprocket teeth >= 9 (smooth mesh)",
            self.sprocket_teeth >= 9,
            f"{self.sprocket_teeth} teeth, r_p = {self.r_pitch_m*1000:.1f} mm",
        ))
        out.append((
            "coil radius >= link min bend radius",
            self.store_footprint_mm / 2.0 >= self.link_min_bend_r_mm,
            f"coil OD {self.store_footprint_mm:.0f} mm vs 2·r_min {2*self.link_min_bend_r_mm:.0f} mm",
        ))
        out.append((
            "engagement fraction in (0,1]",
            0.0 < self.I_engage <= 1.0,
            f"I_engage = {self.I_engage}",
        ))
        return out

    @property
    def is_valid(self) -> bool:
        return all(ok for _, ok, _ in self.checks())


def report(p: ZipChainParams) -> None:
    print("=" * 68)
    print("ZIP-CHAIN LINEAR ACTUATOR  (2-strip rigid chain) — Layer-B model")
    print("=" * 68)
    print(f"  chain pitch ............ {p.pitch_mm:.1f} mm   strands {p.strands}")
    print(f"  meshed column .......... {p.col_w_mm:.0f} x {p.col_h_mm:.0f} mm  "
          f"(I_engage {p.I_engage:.2f})  E_eff {p.E_col_MPa:.0f} MPa")
    print(f"  sprocket ............... z={p.sprocket_teeth}, r_p={p.r_pitch_m*1000:.1f} mm")
    print(f"  E·I column ............. {p.EI:.2f} N·m²")
    print("-" * 68)
    print(f"  deploy speed ........... {p.deploy_speed_mms:.0f} mm/s  @ {p.motor_rpm:.0f} rpm")
    print(f"  drive thrust (τ/r_p) ... {p.F_drive_N:.0f} N  @ {p.motor_torque_Nm:.2f} N·m")
    print(f"  mesh crush cap ......... {p.mesh_crush_N:.0f} N   strand pull cap {p.link_break_N:.0f} N")
    print("-" * 68)
    Lmin = p.retract_free_mm
    Lmax = p.stroke_mm + p.retract_free_mm
    print("  PUSH envelope (buckling-limited):")
    print(f"    at retract  L={Lmin:5.0f} mm : F_cr={p.F_buckle_N(Lmin):7.0f} N -> "
          f"push {p.push_capacity_N(Lmin):6.0f} N")
    print(f"    at full ext L={Lmax:5.0f} mm : F_cr={p.F_buckle_N(Lmax):7.0f} N -> "
          f"push {p.push_capacity_N(Lmax):6.0f} N")
    print(f"    buckling knee L* ..... {p.buckle_knee_mm:.0f} mm  "
          f"({'within stroke — last part is buckling-limited' if p.buckle_knee_mm < Lmax else 'beyond stroke — drive-limited throughout'})")
    print(f"  PULL envelope (tension) : {p.pull_capacity_N(Lmax):.0f} N flat (strand-limited)")
    print(f"  HOLD @ full ext, motor OFF (self-lock): {p.hold_capacity_N(Lmax):.0f} N")
    print("-" * 68)
    print(f"  stored footprint ~ ..... Ø{p.store_footprint_mm:.0f} mm for {p.stroke_mm:.0f} mm stroke "
          f"(store/stroke = {p.compactness_vs_rigid:.2f})")
    print("-" * 68)
    print("  feasibility:")
    for name, ok, detail in p.checks():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name:38s} {detail}")
    print(f"  => {'VALID' if p.is_valid else 'INVALID'}")
    print("=" * 68)


def plot_envelope(p: ZipChainParams) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Lmin = p.retract_free_mm
    Lmax = p.stroke_mm + p.retract_free_mm
    L = np.linspace(Lmin, Lmax, 300)
    strokes = L - Lmin  # 0 at retract, stroke at full extension

    push = np.array([p.push_capacity_N(x) for x in L])
    buckle = np.array([p.F_buckle_N(x) for x in L])
    pull = np.array([p.pull_capacity_N(x) for x in L])
    hold = np.array([p.hold_capacity_N(x) for x in L])

    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.axhline(p.F_drive_N, color="0.6", ls="--", lw=1, label=f"drive ceiling {p.F_drive_N:.0f} N")
    ax.axhline(p.mesh_crush_N, color="0.8", ls=":", lw=1, label=f"mesh crush {p.mesh_crush_N:.0f} N")
    ax.plot(strokes, buckle, color="tab:orange", lw=1.2, ls="-.", label="Euler F_cr(L)")
    ax.plot(strokes, push, color="tab:red", lw=2.6, label="PUSH (usable)")
    ax.plot(strokes, pull, color="tab:blue", lw=2.0, label="PULL (usable)")
    ax.plot(strokes, hold, color="tab:green", lw=1.6, ls="--", label="HOLD, motor off")

    knee = p.buckle_knee_mm - Lmin
    if 0 < knee < (Lmax - Lmin):
        ax.axvline(knee, color="k", lw=0.8, alpha=0.5)
        ax.annotate(f"buckling knee\nL*={p.buckle_knee_mm:.0f} mm",
                    xy=(knee, p.F_drive_N), xytext=(knee + 20, p.F_drive_N * 1.15),
                    fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))

    ax.set_ylim(0, max(p.mesh_crush_N, p.link_break_N) * 1.05)
    ax.set_xlabel("stroke / extension  [mm]")
    ax.set_ylabel("force  [N]")
    ax.set_title("Zip-chain actuator — asymmetric force envelope (push buckling-limited)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = OUT / "envelope.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    p = ZipChainParams()
    report(p)
    out = plot_envelope(p)
    print(f"wrote {out}")
