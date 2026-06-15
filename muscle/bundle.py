"""
1D recruitment bundle: a strand of SMA cells (mujoco/actuator.py's muscle cousin)
pulling one tendon against a bias spring. This is Layer B for the matrix muscle —
the honest thermal + contraction dynamics that any MuJoCo animation or learned
recruitment policy has to sit on top of.

Architecture decision baked in (2026-06-15): HOLDING IS POWERED. There is no
static latch — cut a cell's current and it cools below its transition and loses
force in a few seconds. So the model carries three drive modes per cell:

    OFF   -> no current, cell relaxes toward ambient
    PULSE -> actuate current, overdrives to ~250 C, contracts in ~1 s
    HOLD  -> PWM-reduced current, sits just above the 90 C transition (cheap hold)

The thermal model is first-order, anchored to the datasheet time constants in
SMACellSpec (tau_cool sets the cooling RC; the pulse steady temperature sets the
heating overdrive so contraction lands at ~1 s). Activation is the martensite->
austenite fraction between Mf and Af. Force = activation x recommended pull.
"""

from dataclasses import dataclass, field
from math import sqrt

from cell import SMACellSpec


OFF, PULSE, HOLD = "off", "pulse", "hold"


@dataclass
class ThermalCell:
    """One heated SMA strand element: temperature state + contractile force."""
    spec: SMACellSpec = field(default_factory=SMACellSpec)
    t_ss_pulse: float = 250.0          # transient steady temp at actuate current (overdrive)
    hold_margin_c: float = 5.0         # hold target sits this far above Af

    def __post_init__(self):
        s = self.spec
        self.T = s.ambient_c
        # I^2 R into ONE segment at the recommended (actuate) current
        self.p_pulse = s.rec_current_a ** 2 * s.seg_resistance
        # derive convective loss h and thermal mass C from tau_cool + pulse steady temp
        self.h = self.p_pulse / (self.t_ss_pulse - s.ambient_c)   # W/C
        self.C = self.h * s.tau_cool_s                            # J/C  (tau = C/h)
        # hold: just enough power to sit at Af + margin (the cheap PWM hold)
        self.t_hold = s.austenite_finish_c + self.hold_margin_c
        self.p_hold = self.h * (self.t_hold - s.ambient_c)
        self.i_hold = sqrt(self.p_hold / s.seg_resistance)

    def power(self, mode: str) -> float:
        return {OFF: 0.0, PULSE: self.p_pulse, HOLD: self.p_hold}[mode]

    @property
    def activation(self) -> float:
        """Austenite fraction in [0,1]: 0 below Mf, 1 above Af, linear between."""
        s = self.spec
        lo, hi = s.martensite_finish_c, s.austenite_finish_c
        return min(1.0, max(0.0, (self.T - lo) / (hi - lo)))

    @property
    def force(self) -> float:
        """Contractile force of this strand right now, N."""
        return self.activation * self.spec.pull_force_n

    def step(self, dt: float, mode: str):
        p = self.power(mode)
        self.T += (p - self.h * (self.T - self.spec.ambient_c)) / self.C * dt


@dataclass
class Bundle:
    """n_parallel strands sharing one tendon; series multiplies stroke + heat."""
    spec: SMACellSpec = field(default_factory=SMACellSpec)

    def __post_init__(self):
        self.cells = [ThermalCell(self.spec) for _ in range(self.spec.parallel_count)]

    @property
    def force(self) -> float:
        """Net tendon force: parallel strands sum, bias spring subtracts."""
        gross = sum(c.force for c in self.cells)
        return gross - self.spec.bias_force_n

    def power(self, modes) -> float:
        """Electrical power right now = per-strand power x series segments per strand."""
        return self.spec.series_count * sum(c.power(m) for c, m in zip(self.cells, modes))

    @property
    def peak_temp(self) -> float:
        return max(c.T for c in self.cells)

    def step(self, dt: float, modes):
        for c, m in zip(self.cells, modes):
            c.step(dt, m)


def trace(label, bundle, schedule, dt=0.02):
    """Run a (t_end, modes) schedule, print force/temp/power at phase boundaries."""
    print(f"\n--- {label} ---")
    print(f"  {'t(s)':>5} {'phase':>8} {'force(N)':>9} {'peakT(C)':>9} {'power(W)':>9} {'energy(J)':>10}")
    t, energy = 0.0, 0.0
    for t_end, modes, name in schedule:
        while t < t_end - 1e-9:
            p = bundle.power(modes)
            bundle.step(dt, modes)
            energy += p * dt
            t += dt
        print(f"  {t:5.1f} {name:>8} {bundle.force:9.2f} {bundle.peak_temp:9.0f} "
              f"{bundle.power(modes):9.2f} {energy:10.1f}")
    return energy


if __name__ == "__main__":
    # size a strand for ~10 N / ~10 mm (same 'muscle unit' as cell.py)
    spec = SMACellSpec().size_for(target_force_n=10.0, target_stroke_mm=10.0)
    print(f"bundle: {spec.parallel_count} parallel x {spec.series_count} series, "
          f"net {spec.cell_force_n:.1f} N, stroke {spec.cell_stroke_mm:.1f} mm")
    c0 = ThermalCell(spec)
    print(f"per strand: pulse {c0.p_pulse:.2f} W (-> {c0.t_ss_pulse:.0f}C), "
          f"hold {c0.p_hold:.2f} W @ {c0.i_hold*1000:.0f} mA (sits at {c0.t_hold:.0f}C)")

    n = spec.parallel_count
    all_on = lambda m: [m] * n

    # the lifecycle the architecture decision implies: contract, hold (powered),
    # then release -> force decays, proving there is no static pose.
    b = Bundle(spec)
    e = trace("contract / hold / release (all strands)", b, [
        (1.0, all_on(PULSE), "PULSE"),   # overdrive: contract fast
        (4.0, all_on(HOLD),  "HOLD"),    # PWM hold just above transition
        (6.0, all_on(HOLD),  "HOLD"),
        (6.0, all_on(OFF),   "OFF"),     # cut power...
        (9.0, all_on(OFF),   "OFF"),     # ...force bleeds off as it cools
    ])

    # recruitment summation: force scales with how many strands you fire
    print("\n--- recruitment: force vs strands pulsed (after 1.5 s) ---")
    print(f"  {'strands':>8} {'force(N)':>9} {'power(W)':>9}")
    for k in range(0, n + 1):
        bb = Bundle(spec)
        modes = [PULSE] * k + [OFF] * (n - k)
        for _ in range(int(1.5 / 0.02)):
            bb.step(0.02, modes)
        print(f"  {k:8d} {bb.force:9.2f} {bb.power(modes):9.2f}")

    # hold cost: PWM-hold vs naive full-current hold (the actuate!=hold nuance)
    print("\n--- hold power: PWM-hold vs holding at actuate current ---")
    bh = Bundle(spec)
    for _ in range(int(1.0 / 0.02)):
        bh.step(0.02, all_on(PULSE))
    print(f"  PWM hold ........ {bh.power(all_on(HOLD)):6.2f} W  (sits ~{c0.t_hold:.0f}C)")
    print(f"  full-current hold {bh.power(all_on(PULSE)):6.2f} W  (would cook to {c0.t_ss_pulse:.0f}C)")
