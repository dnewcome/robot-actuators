"""
First-order sizing model of a PIEZO-HYDRAULIC LINEAR ACTUATOR for a robot joint.

This is the linear counterpart to the rotary work in the repo (the cycloidal /
planetary drives and the motor/fea.py torque motor). Where those convert a compact
high-speed *rotary* source into high torque through a clever *gear* transmission,
this converts a compact high-force/tiny-stroke *piezo* source into a high-force
*linear* stroke through a *fluidic* transmission — the fluid does the gearing.

Architecture (piezo-stack piston-pump array  ->  hydraulic cylinder):

    N piezo-stack pump cells  --(one-way valves)-->  sealed cylinder  -->  rod
      kHz, µm stroke, high P                 area ratio = "gear ratio"

Why stacks, not disc benders: a robot joint needs MPa-class pressure for force,
and a piezo *stack* driving a small pump piston is natively a high-pressure /
low-volume source (blocked stress ~30-60 MPa). Disc benders give more stroke but
far less pressure — the low-force / high-strain (soft-muscle) corner, not this one.

The whole point of the model is to expose the three design levers honestly:
  1. VALVE DIODICITY D — flap check-valve (D~50, efficient, fragile/slow) vs.
     etched fluidic diode (D~1.5-2, robust/fast, leaky). Sets rectification eff.
  2. DEAD VOLUME / ENTRAINED AIR (effective bulk modulus β) — each stroke first
     compresses the trapped fluid before the valve opens; air in the oil collapses
     β and kills flow at pressure. This is why stiff fluid + minimal dead volume win.
  3. PISTON AREA — the fluidic "gear ratio". It trades force for speed and, like a
     gear ratio, is conserved in power: peak output power is INDEPENDENT of it.

Everything here is analytical first-order (load-line piezo, linear compressibility,
lumped valve efficiency) — good for sizing and seeing the trade space, NOT a CFD/
transient valve-dynamics model. Trustworthy outputs: the F–v envelope, stall force,
no-load speed, peak power, and the two sensitivity sweeps. The absolute *efficiency*
number is a coarse first-order estimate and is flagged as such.

    ../.venv/bin/python linear/actuator.py            # report + out/envelope.png
"""

from dataclasses import dataclass
from math import pi
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


@dataclass
class PiezoHydraulicParams:
    # --- Pump array ------------------------------------------------------------
    n_cells: int = 6                 # piezo-stack pump cells in parallel (adds flow)
    freq_hz: float = 1000.0          # pumping frequency [Hz] — piezo runs happily at kHz

    # --- Piezo stack (per cell) ------------------------------------------------
    # A co-fired multilayer stack. Two independent material numbers set everything:
    # free strain (-> free stroke) and blocked stress (-> blocked force). Their
    # product times volume is the mechanical energy per stroke (invariant).
    stack_side_mm: float = 10.0      # square stack cross-section side [mm]
    stack_len_mm: float = 20.0       # stack length [mm]
    piezo_free_strain: float = 1.0e-3   # free strain at drive voltage (~0.1%)
    piezo_block_stress_MPa: float = 30.0  # blocked stress [MPa] (working, conservative)
    piezo_conv_eff: float = 0.50     # electrical->delivered-mechanical (coupling k^2 * recovery)

    # --- Pump piston (per cell) — the hydraulic transformer ---------------------
    pump_piston_dia_mm: float = 8.0  # small piston the stack drives; sets Pb vs V0 split

    # --- Fluid / chamber -------------------------------------------------------
    dead_volume_ratio: float = 3.0   # chamber dead volume as multiple of free swept vol
    bulk_modulus_GPa: float = 1.6    # EFFECTIVE bulk modulus β (oil ~1.6; air entrainment
                                     #   drops it by 10-100x — that's the sensitivity sweep)

    # --- Valves ----------------------------------------------------------------
    valve_diodicity: float = 50.0    # D = g_fwd/g_rev. Check valve ~50; fluidic diode ~1.8

    # --- Output cylinder -------------------------------------------------------
    out_piston_dia_mm: float = 16.0  # power-piston bore; the "gear ratio" knob
    stroke_mm: float = 40.0          # usable linear stroke
    seal_eff: float = 0.90           # mechanical/seal efficiency at the rod

    # ---- derived piezo-stack quantities (SI) ----------------------------------
    @property
    def a_stack(self) -> float:                      # stack cross-section [m^2]
        return (self.stack_side_mm * 1e-3) ** 2

    @property
    def free_stroke(self) -> float:                  # δ0, free stack stroke [m]
        return self.piezo_free_strain * self.stack_len_mm * 1e-3

    @property
    def block_force(self) -> float:                  # F_block [N]
        return self.piezo_block_stress_MPa * 1e6 * self.a_stack

    @property
    def a_pump(self) -> float:                        # pump piston area [m^2]
        return pi / 4 * (self.pump_piston_dia_mm * 1e-3) ** 2

    @property
    def block_pressure(self) -> float:                # Pb, blocked pump pressure [Pa]
        return self.block_force / self.a_pump

    @property
    def free_swept_vol(self) -> float:                # V0, free swept vol per stroke [m^3]
        return self.free_stroke * self.a_pump

    @property
    def stroke_energy(self) -> float:                 # piezo mech energy per stroke [J]
        return self.block_force * self.free_stroke    # = Pb * V0 (invariant vs a_pump)

    @property
    def dead_volume(self) -> float:                   # V_dead per cell [m^3]
        return self.dead_volume_ratio * self.free_swept_vol

    @property
    def beta(self) -> float:                          # effective bulk modulus [Pa]
        return self.bulk_modulus_GPa * 1e9

    @property
    def valve_eff(self) -> float:                     # rectification efficiency 1 - 1/D
        return max(0.0, 1.0 - 1.0 / self.valve_diodicity)

    @property
    def a_out(self) -> float:                         # output piston area [m^2]
        return pi / 4 * (self.out_piston_dia_mm * 1e-3) ** 2

    # ---- pump / actuator physics ----------------------------------------------
    def net_swept_vol(self, P):
        """Fluid actually ejected per cell per stroke at chamber pressure P [Pa].

        Piezo load line drops the stroke as pressure rises: V0*(1 - P/Pb).
        Compressibility of the dead volume steals V_dead*P/β before the valve opens.
        Floored at zero (past stall the cell just can't open the valve)."""
        loadline = self.free_swept_vol * (1.0 - P / self.block_pressure)
        compress = self.dead_volume * P / self.beta
        return np.maximum(loadline - compress, 0.0)

    def pump_flow(self, P):
        """Total array volumetric flow at pressure P [m^3/s]."""
        return self.n_cells * self.freq_hz * self.valve_eff * self.net_swept_vol(P)

    @property
    def stall_pressure(self) -> float:
        """Pressure where net swept volume hits zero (v=0) [Pa]."""
        Pb, V0, Vd, beta = self.block_pressure, self.free_swept_vol, self.dead_volume, self.beta
        return Pb / (1.0 + Pb * Vd / (V0 * beta))

    @property
    def stall_force(self) -> float:                   # F at v=0 [N]
        return self.stall_pressure * self.a_out * self.seal_eff

    @property
    def noload_flow(self) -> float:                   # Q at P~0 [m^3/s]
        return self.n_cells * self.freq_hz * self.valve_eff * self.free_swept_vol

    @property
    def noload_speed(self) -> float:                  # v at F=0 [m/s]
        return self.noload_flow / self.a_out

    @property
    def peak_power(self) -> float:
        """Peak hydraulic output power [W]. Envelope is linear -> F_stall*v_noload/4.
        Note: = stall_pressure * noload_flow / 4 — INDEPENDENT of output piston area."""
        return self.stall_force * self.noload_speed / 4.0

    @property
    def sys_efficiency(self) -> float:
        """Coarse first-order electrical->hydraulic efficiency at the peak-power point.
        Product of piezo conversion, valve rectification, seal, and the compressibility
        derating there. FLAGGED as optimistic first-order, not a measured number."""
        P = self.stall_pressure / 2.0                 # peak-power operating pressure
        loadline = self.free_swept_vol * (1.0 - P / self.block_pressure)
        eta_compress = self.net_swept_vol(P) / loadline if loadline > 0 else 0.0
        return self.piezo_conv_eff * self.valve_eff * self.seal_eff * eta_compress

    def fv_curve(self, n=80):
        """Force–velocity envelope. Returns F [N], v [m/s] from no-load to stall."""
        P = np.linspace(0.0, self.stall_pressure, n)
        F = P * self.a_out * self.seal_eff
        v = self.pump_flow(P) / self.a_out
        return F, v

    def pq_curve(self, n=80):
        """Pump pressure–flow curve (whole array). Returns P [Pa], Q [m^3/s], and the
        ideal (incompressible, no dead volume) flow for the compressibility gap."""
        P = np.linspace(0.0, self.block_pressure, n)
        Q = self.pump_flow(P)
        Q_ideal = self.n_cells * self.freq_hz * self.valve_eff * \
            np.maximum(self.free_swept_vol * (1.0 - P / self.block_pressure), 0.0)
        return P, Q, Q_ideal

    # ---- validation -----------------------------------------------------------
    def checks(self):
        c = []
        c.append(("pump piston fits stack face",
                  self.pump_piston_dia_mm < self.stack_side_mm * 1.2,
                  f"Ø{self.pump_piston_dia_mm} vs {self.stack_side_mm}mm face"))
        c.append(("blocked pressure in hydraulic range",
                  self.block_pressure < 70e6,
                  f"{self.block_pressure/1e6:.1f} MPa (< 70 MPa seal/line limit)"))
        c.append(("net flow positive at operating point",
                  self.net_swept_vol(self.stall_pressure / 2.0) > 0,
                  f"{self.net_swept_vol(self.stall_pressure/2)*1e9:.2f} µL/stroke @ half-stall"))
        c.append(("diodicity > 1 (valve actually rectifies)",
                  self.valve_diodicity > 1.0,
                  f"D = {self.valve_diodicity:g}"))
        c.append(("piezo strain physical",
                  self.piezo_free_strain < 3e-3,
                  f"{self.piezo_free_strain*100:.3f}% free strain"))
        c.append(("stall force below buckling of Ø8 rod (~steel)",
                  self.stall_force < pi**3 * 200e9 * (pi/64*(6e-3)**4) / (4*(self.stroke_mm*1e-3)**2),
                  f"F_stall {self.stall_force:.0f} N"))
        c.append(("stroke positive", self.stroke_mm > 0, f"{self.stroke_mm} mm"))
        c.append(("compressibility not dominating (β vs Pb)",
                  self.stall_pressure > 0.5 * self.block_pressure,
                  f"stall {self.stall_pressure/1e6:.1f} of block {self.block_pressure/1e6:.1f} MPa"))
        return c

    @property
    def is_valid(self) -> bool:
        return all(ok for _, ok, _ in self.checks())


def report(p: PiezoHydraulicParams):
    print("  ── piezo-hydraulic linear actuator (first-order sizing) ──")
    print(f"   array            : {p.n_cells} stack-pump cells @ {p.freq_hz:.0f} Hz")
    print(f"   piezo stack/cell : {p.stack_side_mm}×{p.stack_side_mm}×{p.stack_len_mm} mm, "
          f"δ0={p.free_stroke*1e6:.1f} µm, F_block={p.block_force:.0f} N")
    print(f"   pump piston/cell : Ø{p.pump_piston_dia_mm} mm  →  "
          f"Pb={p.block_pressure/1e6:.1f} MPa, V0={p.free_swept_vol*1e9:.2f} µL "
          f"({p.stroke_energy*1e3:.1f} mJ/stroke)")
    print(f"   valves           : D={p.valve_diodicity:g}  →  rectification η={p.valve_eff:.2f} "
          f"({'flap check-valve' if p.valve_diodicity > 10 else 'etched fluidic diode'})")
    print(f"   fluid            : β_eff={p.bulk_modulus_GPa:.2f} GPa, "
          f"dead vol {p.dead_volume_ratio:g}×V0")
    print(f"   output cylinder  : Ø{p.out_piston_dia_mm} mm bore, {p.stroke_mm} mm stroke")
    print("   ---------------------------------------------------------")
    print(f"   STALL FORCE      ≈ {p.stall_force:.0f} N   ({p.stall_force/9.81:.0f} kgf, "
          f"@ {p.stall_pressure/1e6:.1f} MPa)")
    print(f"   NO-LOAD SPEED    ≈ {p.noload_speed*1e3:.1f} mm/s  "
          f"({p.noload_flow*6e7:.0f} mL/min through Ø{p.out_piston_dia_mm})")
    print(f"   full stroke time ≈ {p.stroke_mm*1e-3/p.noload_speed:.2f} s (no load)")
    print(f"   PEAK POWER       ≈ {p.peak_power:.0f} W  (invariant of piston area)")
    print(f"   sys efficiency   ≈ {p.sys_efficiency*100:.0f}%  (first-order, optimistic)")
    print(f"   validity         : {'ALL CHECKS PASS' if p.is_valid else 'FAILING CHECKS'}")


def render(p: PiezoHydraulicParams, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Piezo-hydraulic linear actuator — first-order envelope & sensitivities",
                 fontsize=13, weight="bold")

    # (a) Force–velocity envelope — the actuator's "torque-speed" analog ---------
    F, v = p.fv_curve()
    a = ax[0, 0]
    a.fill_between(F, 0, v * 1e3, alpha=0.15, color="tab:blue")
    a.plot(F, v * 1e3, color="tab:blue", lw=2)
    # peak-power point (midpoint of the linear envelope)
    Fp, vp = p.stall_force / 2, p.noload_speed / 2
    a.plot([Fp], [vp * 1e3], "o", color="tab:red")
    a.annotate(f"peak {p.peak_power:.0f} W", (Fp, vp * 1e3),
               textcoords="offset points", xytext=(6, 6), color="tab:red", fontsize=9)
    a.plot([p.stall_force], [0], "s", color="k")
    a.annotate(f"stall {p.stall_force:.0f} N", (p.stall_force, 0),
               textcoords="offset points", xytext=(-10, 8), ha="right", fontsize=9)
    a.plot([0], [p.noload_speed * 1e3], "^", color="tab:green")
    a.annotate(f"{p.noload_speed*1e3:.0f} mm/s", (0, p.noload_speed * 1e3),
               textcoords="offset points", xytext=(8, -2), fontsize=9)
    a.set_xlabel("output force  F  [N]")
    a.set_ylabel("output speed  v  [mm/s]")
    a.set_title("(a) force–velocity envelope", fontsize=10)
    a.grid(alpha=0.3)

    # (b) Pump P–Q with the compressibility gap ----------------------------------
    P, Q, Qi = p.pq_curve()
    a = ax[0, 1]
    a.plot(P / 1e6, Qi * 6e7, "--", color="gray", lw=1.5, label="ideal (stiff, no dead vol)")
    a.fill_between(P / 1e6, Q * 6e7, Qi * 6e7, alpha=0.15, color="tab:orange")
    a.plot(P / 1e6, Q * 6e7, color="tab:orange", lw=2, label="actual (β, dead vol)")
    a.axvline(p.stall_pressure / 1e6, color="k", ls=":", lw=1)
    a.set_xlabel("pump pressure  P  [MPa]")
    a.set_ylabel("array flow  Q  [mL/min]")
    a.set_title("(b) pump curve — gap = compressibility loss", fontsize=10)
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    # (c) Sensitivity to VALVE DIODICITY -----------------------------------------
    a = ax[1, 0]
    D = np.geomspace(1.2, 100, 60)
    pk, fl = [], []
    for d in D:
        q = PiezoHydraulicParams(**{**p.__dict__, "valve_diodicity": float(d)})
        pk.append(q.peak_power)
        fl.append(q.noload_flow * 6e7)
    a.semilogx(D, pk, color="tab:blue", lw=2)
    a.set_xlabel("valve diodicity  D  (g_fwd / g_rev)")
    a.set_ylabel("peak power  [W]", color="tab:blue")
    a.axvspan(1.3, 2.2, alpha=0.12, color="tab:red")
    a.axvspan(20, 100, alpha=0.12, color="tab:green")
    a.text(1.7, a.get_ylim()[1] * 0.12, "etched\nfluidic\ndiode", ha="center",
           fontsize=8, color="tab:red")
    a.text(45, a.get_ylim()[1] * 0.12, "flap\ncheck\nvalve", ha="center",
           fontsize=8, color="tab:green")
    a.set_title("(c) valve trade: leaky-but-robust vs. efficient-but-fragile", fontsize=10)
    a.grid(alpha=0.3, which="both")

    # (d) Sensitivity to EFFECTIVE BULK MODULUS (entrained air) -------------------
    a = ax[1, 1]
    beta = np.geomspace(0.03, 1.8, 60)   # GPa: lots of air -> stiff oil
    st = []
    for b in beta:
        q = PiezoHydraulicParams(**{**p.__dict__, "bulk_modulus_GPa": float(b)})
        st.append(q.stall_force)
    a.semilogx(beta, st, color="tab:purple", lw=2)
    a.axvline(p.bulk_modulus_GPa, color="k", ls=":", lw=1)
    a.set_xlabel("effective bulk modulus  β  [GPa]   (← more entrained air)")
    a.set_ylabel("stall force  [N]", color="tab:purple")
    a.set_title("(d) why stiff fluid matters — air collapses force", fontsize=10)
    a.grid(alpha=0.3, which="both")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.mkdir(exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    p = PiezoHydraulicParams()
    print("Piezo-hydraulic linear actuator — robot-joint sizing\n")
    report(p)
    print("\n  validation:")
    for name, ok, detail in p.checks():
        print(f"    [{'ok' if ok else 'XX'}] {name:44s} {detail}")
    render(p, OUT / "envelope.png")
    print(f"\n  wrote {OUT/'envelope.png'}")
