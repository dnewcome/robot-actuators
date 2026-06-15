"""
The UNIT CELL of the matrix muscle: one SMA (Nitinol) contractile element.

This is the muscle analogue of mujoco/actuator.py — a spec-as-code that turns a
real, buyable wire into the force / stroke / timing / power numbers the matrix
sim (Layer A) and the recruitment-learning layer have to honor. The cell is
deliberately DUMB: drive = "current on -> Joule heat -> contract". All the
intelligence lives upstream in how many cells fire, in what pattern.

Numbers anchor to Dynalloy Flexinol-90, 0.006" (150 um) — published datasheet
figures, so the sim models an actual wire, not a fantasy actuator.

Architecture note (the sarcomere mapping):
    parallel wires  -> forces ADD,   stroke fixed   (myofibrils => strength)
    series segments -> strokes ADD,  force fixed     (sarcomeres => range)
SMA's ~4% strain is the whole design tension: usable stroke FORCES series
stacking (and a lot of wire length), which is exactly the muscle architecture.
"""

from dataclasses import dataclass
from math import pi, ceil


@dataclass
class SMACellSpec:
    """One SMA contractile cell. Defaults = Flexinol-90, 0.006" (150 um)."""

    # --- wire (Dynalloy Flexinol-90, 0.006") -----------------------------
    wire_dia_um: float = 150.0          # 0.006"
    resistance_per_m: float = 40.0      # ohm/m  (datasheet)
    rec_stress_mpa: float = 172.0       # recommended pull stress for long cycle life
    max_stress_mpa: float = 600.0       # ultimate; do not approach in service
    usable_strain: float = 0.04         # recoverable strain at rec. cycle life (4%)

    # --- thermal / transition (90 C alloy) -------------------------------
    austenite_finish_c: float = 90.0    # fully CONTRACTED above this (As..Af)
    martensite_finish_c: float = 62.0   # fully RELAXED below this; hysteresis ~28 C
    ambient_c: float = 20.0
    tau_heat_s: float = 1.0             # contraction time at recommended current
    tau_cool_s: float = 2.7             # passive convective relaxation (still air)
    rec_current_a: float = 0.41         # recommended activation current
    drive_voltage_max: float = 12.0     # per-cell supply ceiling (sets max series length)

    # --- cell packaging (the sarcomere architecture) ---------------------
    active_length_mm: float = 50.0      # one wire SEGMENT's active (contracting) length
    series_count: int = 1               # segments mechanically in series -> stroke x
    parallel_count: int = 1             # wires mechanically in parallel -> force x
    bias_fraction: float = 0.5          # return-spring force as a fraction of 1-wire pull

    # ---- single-wire physics ------------------------------------------------
    @property
    def area_m2(self) -> float:
        r = self.wire_dia_um * 1e-6 / 2.0
        return pi * r * r

    @property
    def pull_force_n(self) -> float:
        """Recommended contractile force of ONE wire (rec_stress x area)."""
        return self.rec_stress_mpa * 1e6 * self.area_m2

    @property
    def seg_resistance(self) -> float:
        """Electrical resistance of one active segment, ohm."""
        return self.resistance_per_m * (self.active_length_mm / 1000.0)

    @property
    def single_stroke_mm(self) -> float:
        """Contraction of ONE segment = usable_strain x active length."""
        return self.usable_strain * self.active_length_mm

    # ---- packaged cell (series x parallel) ----------------------------------
    @property
    def cell_force_n(self) -> float:
        """Net contractile force the packaged cell delivers, bias subtracted.
        Parallel multiplies force; the return spring (bias) opposes it."""
        gross = self.pull_force_n * self.parallel_count
        return gross - self.bias_force_n

    @property
    def bias_force_n(self) -> float:
        """Return-spring force (resets the cooled wire). Scales with parallel count."""
        return self.bias_fraction * self.pull_force_n * self.parallel_count

    @property
    def cell_stroke_mm(self) -> float:
        """Series multiplies stroke."""
        return self.single_stroke_mm * self.series_count

    @property
    def n_segments(self) -> int:
        return self.series_count * self.parallel_count

    @property
    def wire_total_mm(self) -> float:
        """Total wire in the cell — SMA's hidden cost: stroke is bought by length."""
        return self.active_length_mm * self.n_segments

    # ---- electrical / thermal ----------------------------------------------
    @property
    def hold_power_w(self) -> float:
        """Power to keep the WHOLE cell hot/contracted: I^2 R over all segments."""
        return self.n_segments * self.rec_current_a ** 2 * self.seg_resistance

    @property
    def series_drive_voltage(self) -> float:
        """If a series strand is wired electrically in series too: I*R*series."""
        return self.rec_current_a * self.seg_resistance * self.series_count

    @property
    def bandwidth_hz(self) -> float:
        """Full contract+relax cycle rate — cooling-limited, sets the sim timescale."""
        return 1.0 / (self.tau_heat_s + self.tau_cool_s)

    @property
    def duty_ratio(self) -> float:
        """Fraction of a cycle the wire is being actively heated (the rest is cooling)."""
        return self.tau_heat_s / (self.tau_heat_s + self.tau_cool_s)

    # ---- sizing helper ------------------------------------------------------
    def size_for(self, target_force_n: float, target_stroke_mm: float) -> "SMACellSpec":
        """Return a copy packaged to meet a force & stroke target. Force -> parallel,
        stroke -> series. Exposes how much wire a 'muscle unit' really costs."""
        gross_per_wire = self.pull_force_n * (1.0 - self.bias_fraction)
        n_par = max(1, ceil(target_force_n / gross_per_wire))
        n_ser = max(1, ceil(target_stroke_mm / self.single_stroke_mm))
        from dataclasses import replace
        return replace(self, parallel_count=n_par, series_count=n_ser)

    def report(self):
        print("\n=== SMA unit cell (Flexinol-90, %.0f um) ===" % self.wire_dia_um)
        print(f"single wire ...... {self.pull_force_n:.2f} N pull @ {self.rec_stress_mpa:.0f} MPa, "
              f"{self.usable_strain*100:.0f}% strain, R={self.resistance_per_m:.0f} ohm/m")
        print(f"segment ({self.active_length_mm:.0f} mm) "
              f"stroke {self.single_stroke_mm:.2f} mm   R={self.seg_resistance:.2f} ohm   "
              f"I_rec={self.rec_current_a:.2f} A")
        print(f"thermal .......... contract {self.tau_heat_s:.1f}s / relax {self.tau_cool_s:.1f}s "
              f"-> {self.bandwidth_hz:.2f} Hz, duty {self.duty_ratio*100:.0f}% heating")
        print(f"transition ....... contract >{self.austenite_finish_c:.0f}C, "
              f"relax <{self.martensite_finish_c:.0f}C (needs bias spring to reset)")
        print(f"\npackaged cell: {self.parallel_count} parallel x {self.series_count} series "
              f"= {self.n_segments} segments, {self.wire_total_mm/1000:.2f} m wire")
        print(f"  force .......... {self.cell_force_n:.2f} N net "
              f"(bias spring takes {self.bias_force_n:.2f} N)")
        print(f"  stroke ......... {self.cell_stroke_mm:.2f} mm  ({self.cell_stroke_mm/(self.active_length_mm*self.series_count)*100:.0f}% of strand)")
        print(f"  hold power ..... {self.hold_power_w:.2f} W  (all segments hot, I^2R)")
        print(f"  series volts ... {self.series_drive_voltage:.2f} V if strand wired in series")
        print()


def matrix_budget(cell: SMACellSpec, rows: int, cols: int):
    """What a rows x cols sheet of these cells costs — the matrix-level reality
    the recruitment controller has to live inside (power, wire, worst-case heat)."""
    n = rows * cols
    print(f"=== {rows}x{cols} matrix = {n} cells ===")
    print(f"  peak force (all parallel) ... {cell.cell_force_n * rows:.1f} N "
          f"(if {rows} rows recruited in parallel)")
    print(f"  max stroke (all series) ..... {cell.cell_stroke_mm * cols:.1f} mm")
    print(f"  total wire .................. {cell.wire_total_mm * n / 1000:.1f} m")
    print(f"  worst-case hold power ....... {cell.hold_power_w * n:.0f} W (every cell hot)")
    print(f"  -> thermal crosstalk + this power ceiling are the real constraints the")
    print(f"     learned recruitment policy must respect (don't hold the whole sheet hot).\n")


if __name__ == "__main__":
    base = SMACellSpec()
    base.report()

    # a finger-muscle-scale unit: ~10 N, ~10 mm stroke — see the wire cost of SMA's low strain
    unit = base.size_for(target_force_n=10.0, target_stroke_mm=10.0)
    print("--- sized for 10 N / 10 mm (a 'muscle unit') ---")
    unit.report()

    matrix_budget(unit, rows=4, cols=4)
