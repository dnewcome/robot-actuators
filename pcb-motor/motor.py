"""
First-order model of an AXIAL-FLUX CORELESS PCB-STATOR MOTOR — the "printed motor" (the
MotorCell paradigm) — and a REDUCTION-NEED CALCULATOR that says whether a CoreXY axis can be
direct-driven or needs a belt/capstan/planetary.

Topology: flat spiral coils etched across the PCB copper layers are the stator; a rotor of
axial magnets (a back-iron or Halbach disc) spins across a small air gap. The radial coil
runs cross the axial gap flux B_g, so current I makes a tangential Lorentz force → torque.
No iron in the stator → NO cogging/detent (why it must be a SERVO, not a stepper) and no
saturation, so torque is linear in current: T = Kt·I.

The honest story this model exposes: Kt can be respectable, but the traces are THIN, so phase
resistance is high and I²R heating caps the CONTINUOUS current hard. PCB motors are therefore
high-Kt / high-R → only ~mN·m CONTINUOUS (more in short peaks). That limits speed/accel, NOT
precision — precision is the load-side encoder's job (see corexy/, linear-rail-servo/).

Model chain (all first-order, calibratable by k_w against a real motor like MotorCell):
    turns  = k_pack·(r_out−r_in)/pitch · layers          (spiral packing)
    Kt     = k_w · N_coils · turns · 2 · B_g · L_a · r_m  (Lorentz sum over radial runs)
    R_phase= ρ_cu · L_wire / (w_trace · t_cu)             (thin copper → high R)
    Km     = Kt / √R_phase                                (geometry-fair figure of merit)
    I_cont = √(P_therm / R_phase)                         (natural convection sets it)
    T_cont = Kt · I_cont ,   T_peak = Kt · I_peak (driver/voltage limited, short bursts)

Trustworthy: the SCALING and the trade-offs (thin copper → thermal wall; more layers help Kt
and R together so Km barely moves; direct-drive vs reduction from the calculator). Absolute
torque is calibrated via k_w — benchmark against MotorCell before trusting the number.

    ../.venv/bin/python pcb-motor/motor.py            # report + out/motor.png
"""

from dataclasses import dataclass
from math import pi, sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

RHO_CU = 1.72e-8          # copper resistivity [Ω·m]
G = 9.81


@dataclass
class PCBMotorParams:
    # --- stator coil geometry --------------------------------------------------
    r_out_mm: float = 15.0        # coil outer radius
    r_in_mm: float = 6.0          # coil inner radius
    n_coils: int = 12             # stator coils (concentrated)
    n_layers: int = 6             # PCB copper layers (coils stacked in series)
    t_cu_um: float = 35.0         # copper thickness per layer (1 oz = 35 µm)
    trace_w_mm: float = 0.15      # trace width
    trace_gap_mm: float = 0.15    # trace-to-trace gap
    k_pack: float = 0.6           # spiral packing efficiency (turns realised vs radial space)
    k_w: float = 0.35             # winding/alignment factor (calibrate vs a real motor)

    # --- rotor magnets ---------------------------------------------------------
    n_poles: int = 14             # rotor pole count (12N14P-ish)
    b_gap_T: float = 0.25         # axial flux density at the coils (gap + magnet dependent)
    max_rpm: float = 4000.0       # mechanical speed ceiling (bearing / back-EMF)

    # --- drive / thermal -------------------------------------------------------
    i_peak_A: float = 1.5         # short-burst current (driver/voltage limited)
    amb_C: float = 25.0
    max_coil_C: float = 90.0      # copper temp limit
    h_conv: float = 12.0          # natural-convection coeff [W/m²K] (both board faces)
    n_phases: int = 3

    # ---- geometry (SI) --------------------------------------------------------
    @property
    def r_out(self): return self.r_out_mm * 1e-3
    @property
    def r_in(self): return self.r_in_mm * 1e-3
    @property
    def L_a(self): return self.r_out - self.r_in          # active radial length
    @property
    def r_m(self): return 0.5 * (self.r_out + self.r_in)
    @property
    def pitch(self): return (self.trace_w_mm + self.trace_gap_mm) * 1e-3

    @property
    def turns_per_coil(self) -> float:
        """Total series turns in one coil (all layers)."""
        per_layer = self.k_pack * self.L_a / self.pitch
        return per_layer * self.n_layers

    # ---- torque constant ------------------------------------------------------
    @property
    def kt(self) -> float:
        """Torque per amp [N·m/A].  Lorentz sum over the two radial runs of every turn of
        every coil, de-rated by the winding factor."""
        return self.k_w * self.n_coils * self.turns_per_coil * 2 * self.b_gap_T * self.L_a * self.r_m

    @property
    def kv_rpm_per_V(self) -> float:
        """Back-EMF: Kv [rpm/V] = 60/(2π·Kt)  (Kt in V·s/rad = N·m/A)."""
        return 60.0 / (2 * pi * self.kt)

    # ---- resistance & motor constant ------------------------------------------
    @property
    def turn_perimeter(self) -> float:
        """Mean length of one turn: two radial runs + two tangential runs at r_m."""
        tang = 2 * pi * self.r_m / self.n_coils          # one coil's angular span at r_m
        return 2 * self.L_a + 2 * tang

    @property
    def r_phase(self) -> float:
        """Per-phase resistance [Ω]. Coils split across phases, series within a phase."""
        coils_per_phase = self.n_coils / self.n_phases
        L_wire = coils_per_phase * self.turns_per_coil * self.turn_perimeter
        a_cu = self.trace_w_mm * 1e-3 * self.t_cu_um * 1e-6
        return RHO_CU * L_wire / a_cu

    @property
    def km(self) -> float:
        """Motor constant Kt/√R [N·m/√W] — the geometry-fair torque-per-heat figure."""
        return self.kt / sqrt(self.r_phase)

    # ---- thermal-limited continuous, and peak ---------------------------------
    @property
    def board_area(self) -> float:
        return 2 * pi * (self.r_out**2 - self.r_in**2)   # both faces of the coil annulus

    @property
    def p_thermal(self) -> float:
        """Continuous copper loss the board can shed by natural convection [W]."""
        return self.h_conv * self.board_area * (self.max_coil_C - self.amb_C)

    @property
    def i_cont(self) -> float:
        return sqrt(self.p_thermal / self.r_phase)

    @property
    def t_cont(self) -> float:
        return self.kt * self.i_cont

    @property
    def t_peak(self) -> float:
        return self.kt * self.i_peak_A

    @property
    def omega_max(self) -> float:
        return self.max_rpm * 2 * pi / 60.0

    # ---- validation -----------------------------------------------------------
    def checks(self):
        c = []
        c.append(("coil annulus sane", self.r_out > self.r_in > 0,
                  f"r {self.r_in_mm}–{self.r_out_mm} mm"))
        c.append(("turns realisable", self.turns_per_coil > 3,
                  f"{self.turns_per_coil:.0f} series turns/coil"))
        c.append(("pole/coil combo valid", self.n_poles != self.n_coils and self.n_poles % 2 == 0,
                  f"{self.n_coils}coil / {self.n_poles}pole"))
        c.append(("continuous current below peak", self.i_cont < self.i_peak_A,
                  f"I_cont {self.i_cont:.2f} < I_peak {self.i_peak_A} A"))
        c.append(("phase resistance not absurd", self.r_phase < 200,
                  f"{self.r_phase:.1f} Ω/phase (thin copper)"))
        c.append(("continuous torque in PCB-motor range", 0.1e-3 < self.t_cont < 60e-3,
                  f"{self.t_cont*1e3:.1f} mN·m continuous"))
        return c

    @property
    def is_valid(self) -> bool:
        return all(ok for _, ok, _ in self.checks())


# --------------------------------------------------------------------------- #
# REDUCTION-NEED CALCULATOR
# --------------------------------------------------------------------------- #
@dataclass
class AxisSpec:
    gantry_mass_kg: float = 0.20     # moving mass on this axis
    pulley_r_mm: float = 6.0         # drive pulley radius (direct-drive)
    target_accel_g: float = 0.5      # wanted acceleration [g]
    target_speed_ms: float = 0.20    # wanted top speed [m/s]
    process_force_N: float = 0.0     # cutting/drawing force at the tool


def reduction_advice(mot: PCBMotorParams, ax: AxisSpec):
    """Given the motor and an axis spec, find the reduction ratio N that meets the peak-accel
    and top-speed targets, and say whether direct-drive works."""
    r = ax.pulley_r_mm * 1e-3
    need_force = ax.gantry_mass_kg * ax.target_accel_g * G + ax.process_force_N

    def accel_g(N, torque):        # gantry accel [g] at reduction N
        return (torque * N / r) / ax.gantry_mass_kg / G

    def speed(N):                  # top speed [m/s] at reduction N
        return (mot.omega_max / N) * r

    # smallest N (>=1) whose PEAK force clears the target and whose speed still meets target
    Ns = np.arange(1.0, 60.0, 0.25)
    ok = [(accel_g(N, mot.t_peak) >= ax.target_accel_g and speed(N) >= ax.target_speed_ms)
          for N in Ns]
    N_rec = next((N for N, o in zip(Ns, ok) if o), None)
    return dict(need_force=need_force,
                direct_force_peak=mot.t_peak / r,
                direct_force_cont=mot.t_cont / r,
                direct_accel_peak=accel_g(1.0, mot.t_peak),
                direct_accel_cont=accel_g(1.0, mot.t_cont),
                direct_speed=speed(1.0),
                N_rec=N_rec, Ns=Ns, accel_g=accel_g, speed=speed)


def report(mot: PCBMotorParams | None = None):
    mot = mot or PCBMotorParams()
    print("\n" + "=" * 68)
    print("  AXIAL-FLUX CORELESS PCB MOTOR  —  first-order model")
    print("=" * 68)
    print(f"  coils/poles            {mot.n_coils}N / {mot.n_poles}P, "
          f"{mot.n_layers} layers × {mot.t_cu_um:.0f} µm cu")
    print(f"  coil annulus           Ø{2*mot.r_in_mm:.0f}–Ø{2*mot.r_out_mm:.0f} mm "
          f"(active L {mot.L_a*1e3:.1f} mm), {mot.turns_per_coil:.0f} turns/coil")
    print(f"  air-gap flux B_g       {mot.b_gap_T:.2f} T")
    print("-" * 68)
    print(f"  Kt                     {mot.kt*1e3:.2f} mN·m/A   (Kv {mot.kv_rpm_per_V:.0f} rpm/V)")
    print(f"  phase resistance R     {mot.r_phase:.1f} Ω        (thin traces!)")
    print(f"  motor constant Km      {mot.km*1e3:.2f} mN·m/√W")
    print("-" * 68)
    print(f"  thermal budget         {mot.p_thermal:.2f} W  →  I_cont {mot.i_cont:.2f} A")
    print(f"  CONTINUOUS torque      {mot.t_cont*1e3:.1f} mN·m")
    print(f"  PEAK torque (@{mot.i_peak_A} A)   {mot.t_peak*1e3:.1f} mN·m")
    print(f"  max speed              {mot.max_rpm:.0f} rpm")
    print("-" * 68)
    for name, ok, detail in mot.checks():
        print(f"    [{'ok ' if ok else 'XX '}] {name:<38} {detail}")
    print(f"  -> {'VALID' if mot.is_valid else 'INVALID'}")
    print("-" * 68)
    ax = AxisSpec()
    adv = reduction_advice(mot, ax)
    print(f"  REDUCTION for a {ax.gantry_mass_kg*1e3:.0f} g gantry, "
          f"{ax.target_accel_g:.1f} g @ {ax.target_speed_ms*1e3:.0f} mm/s target:")
    print(f"    direct-drive: peak {adv['direct_accel_peak']:.2f} g / "
          f"cont {adv['direct_accel_cont']:.2f} g, top {adv['direct_speed']:.2f} m/s")
    if adv["N_rec"] is None:
        print("    -> even geared it can't hit both targets — bigger motor or lighter gantry")
    elif adv["N_rec"] <= 1.0:
        print("    -> DIRECT DRIVE works (no reduction needed)")
    else:
        print(f"    -> needs ~{adv['N_rec']:.1f}:1 reduction (a belt/capstan stage, not a planetary)")
    print("=" * 68 + "\n")
    return mot


def render(mot: PCBMotorParams | None = None):
    mot = mot or PCBMotorParams()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9))

    # (a) torque vs current, with the thermal-continuous and peak points
    I = np.linspace(0, mot.i_peak_A * 1.1, 100)
    ax[0, 0].plot(I, mot.kt * I * 1e3, color="C0", lw=1.8)
    ax[0, 0].axvspan(mot.i_cont, mot.i_peak_A * 1.1, color="C3", alpha=0.10)
    ax[0, 0].scatter([mot.i_cont], [mot.t_cont * 1e3], color="C2", zorder=5,
                     label=f"continuous {mot.t_cont*1e3:.1f} mN·m")
    ax[0, 0].scatter([mot.i_peak_A], [mot.t_peak * 1e3], color="C3", zorder=5,
                     label=f"peak {mot.t_peak*1e3:.1f} mN·m")
    ax[0, 0].set_title("(a) T = Kt·I (linear, no iron) — I²R thermal caps continuous")
    ax[0, 0].set_xlabel("phase current  [A]"); ax[0, 0].set_ylabel("torque  [mN·m]")
    ax[0, 0].legend(fontsize=8, loc="upper left"); ax[0, 0].grid(alpha=0.3)

    # (b) the layer trap: adding copper layers lifts Kt AND R together → Km barely moves
    layers = np.arange(2, 17, 2)
    kt_l, tcont_l, km_l = [], [], []
    for nl in layers:
        m = PCBMotorParams(**{**mot.__dict__, "n_layers": int(nl)})
        kt_l.append(m.kt * 1e3); tcont_l.append(m.t_cont * 1e3); km_l.append(m.km * 1e3)
    ax[0, 1].plot(layers, kt_l, "o-", color="C0", label="Kt [mN·m/A]")
    ax[0, 1].plot(layers, tcont_l, "o-", color="C2", label="T_cont [mN·m]")
    axk = ax[0, 1].twinx()
    axk.plot(layers, km_l, "s--", color="C4", label="Km [mN·m/√W]")
    axk.set_ylabel("Km  [mN·m/√W]", color="C4"); axk.tick_params(axis="y", labelcolor="C4")
    ax[0, 1].axvline(mot.n_layers, color="k", ls=":", lw=0.8)
    ax[0, 1].set_title("(b) layers: Kt grows ∝N, but R climbs too → T_cont only ∝√N")
    ax[0, 1].set_xlabel("copper layers"); ax[0, 1].set_ylabel("Kt / T_cont")
    ax[0, 1].legend(fontsize=8, loc="upper left"); ax[0, 1].grid(alpha=0.3)

    # (c) THE reduction-need chart: gantry accel vs reduction ratio
    ax_spec = AxisSpec()
    adv = reduction_advice(mot, ax_spec)
    Ns = adv["Ns"]
    ap = np.array([adv["accel_g"](N, mot.t_peak) for N in Ns])
    ac = np.array([adv["accel_g"](N, mot.t_cont) for N in Ns])
    sp = np.array([adv["speed"](N) for N in Ns])
    ax[1, 0].plot(Ns, ap, color="C3", lw=1.8, label="peak accel")
    ax[1, 0].plot(Ns, ac, color="C2", lw=1.8, label="continuous accel")
    ax[1, 0].axhline(ax_spec.target_accel_g, color="k", ls="--", lw=0.9, label="target accel")
    # shade where top speed drops below target (too much reduction)
    too_slow = sp < ax_spec.target_speed_ms
    if too_slow.any():
        ax[1, 0].axvspan(Ns[too_slow][0], Ns[-1], color="grey", alpha=0.15, label="too slow (speed)")
    if adv["N_rec"] is not None:
        ax[1, 0].axvline(adv["N_rec"], color="C0", lw=1.2)
        ax[1, 0].annotate(f"  ~{adv['N_rec']:.0f}:1", (adv["N_rec"], ax_spec.target_accel_g),
                          color="C0", fontsize=9)
    ax[1, 0].set_title(f"(c) reduction need: {ax_spec.gantry_mass_kg*1e3:.0f} g gantry, "
                       f"{ax_spec.target_accel_g:.1f} g @ {ax_spec.target_speed_ms*1e3:.0f} mm/s")
    ax[1, 0].set_xlabel("reduction ratio N:1"); ax[1, 0].set_ylabel("achievable accel  [g]")
    ax[1, 0].set_ylim(0, max(6 * ax_spec.target_accel_g, adv["direct_accel_peak"] * 1.4))
    ax[1, 0].legend(fontsize=8, loc="upper right"); ax[1, 0].grid(alpha=0.3)

    # (d) continuous torque + Km vs air-gap flux (magnet/gap lever)
    Bs = np.linspace(0.12, 0.45, 40)
    tc = [PCBMotorParams(**{**mot.__dict__, "b_gap_T": b}).t_cont * 1e3 for b in Bs]
    ax[1, 1].plot(Bs, tc, color="C2", lw=1.8)
    ax[1, 1].scatter([mot.b_gap_T], [mot.t_cont * 1e3], color="C2", zorder=5)
    ax[1, 1].set_title("(d) continuous torque ∝ gap flux (Halbach / smaller gap)")
    ax[1, 1].set_xlabel("air-gap flux B_g  [T]"); ax[1, 1].set_ylabel("T_cont  [mN·m]")
    ax[1, 1].grid(alpha=0.3)

    fig.suptitle(f"Axial-flux PCB motor — {mot.t_cont*1e3:.1f} mN·m continuous / "
                 f"{mot.t_peak*1e3:.0f} mN·m peak, high-speed low-torque (thermal-walled)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "motor.png", dpi=120)
    print(f"  wrote {OUT/'motor.png'}")


if __name__ == "__main__":
    m = report()
    render(m)
