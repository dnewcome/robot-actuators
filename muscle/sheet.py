"""
2D matrix muscle with thermal crosstalk + recruitment rotation.

This is the payoff of the "holding is powered" decision (see matrix-muscle memory):
since a held cell sits hot continuously and there is no latch, the controller's
real job is THERMAL LOAD-BALANCING — spread the holding duty across the sheet so
no single cell is pinned hot forever (the SMA fatigue analogue of biological
motor-unit rotation).

Two physics additions over bundle.py:
  1. Lateral heat conduction between 4-neighbors (crosstalk): a held cell warms
     its neighbors, so cells are NOT thermally independent. coupling g = ratio*h.
  2. Sheet mechanics: C parallel columns, each R cells in SERIES. A series chain's
     force is set by its WEAKEST (least-activated) link — so a whole column must be
     hot together; the rotatable unit is a column. Columns sum to the tendon force.

Cells are dumb: the controller picks OFF/PULSE/HOLD per column, no per-cell temp
feedback. So clustered/edge effects from crosstalk are real and uncorrected —
exactly what a learned recruitment policy would have to manage.

Compares naive (fixed columns) vs rotation on: force tracking, peak temperature,
and MAX PER-CELL HOT-DUTY (the fatigue proxy rotation is meant to cut).
"""

from dataclasses import dataclass, field

from cell import SMACellSpec
from bundle import ThermalCell, OFF, PULSE, HOLD


@dataclass
class Sheet:
    rows: int = 3                  # series per column -> stroke
    cols: int = 8                  # parallel columns -> force + rotation headroom
    spec: SMACellSpec = field(default_factory=SMACellSpec)
    coupling_ratio: float = 0.4    # lateral conductance g as a fraction of convective h
    hold_threshold: float = 0.95   # activation above which a column switches PULSE->HOLD

    def __post_init__(self):
        self.cells = [[ThermalCell(self.spec) for _ in range(self.cols)]
                      for _ in range(self.rows)]
        self.g = self.coupling_ratio * self.cells[0][0].h
        self.duty = [[0.0] * self.cols for _ in range(self.rows)]  # seconds held hot

    def _neighbors(self, r, c):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if 0 <= r + dr < self.rows and 0 <= c + dc < self.cols:
                yield self.cells[r + dr][c + dc]

    def col_activation(self, c) -> float:
        """Series chain force is set by its weakest link -> min activation."""
        return min(self.cells[r][c].activation for r in range(self.rows))

    @property
    def force(self) -> float:
        """Tendon force = sum of column contractile forces (parallel columns)."""
        return sum(self.col_activation(c) for c in range(self.cols)) * self.spec.pull_force_n

    @property
    def peak_temp(self) -> float:
        return max(cell.T for row in self.cells for cell in row)

    def col_modes(self, wanted) -> dict:
        """Per-column drive: pulse a wanted-but-cold column, hold it once hot, else off."""
        modes = {}
        for c in range(self.cols):
            if c in wanted:
                modes[c] = HOLD if self.col_activation(c) >= self.hold_threshold else PULSE
            else:
                modes[c] = OFF
        return modes

    def step(self, dt, modes, amb=None):
        """Synchronous update: conduction uses the current temps, then all advance."""
        amb = self.spec.ambient_c if amb is None else amb
        new_T = [[0.0] * self.cols for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.cells[r][c]
                p_in = cell.power(modes[c])
                conv = cell.h * (cell.T - amb)
                cond = self.g * sum(cell.T - nb.T for nb in self._neighbors(r, c))
                new_T[r][c] = cell.T + (p_in - conv - cond) / cell.C * dt
                if cell.activation > 0.5:
                    self.duty[r][c] += dt
        for r in range(self.rows):
            for c in range(self.cols):
                self.cells[r][c].T = new_T[r][c]

    def max_duty_frac(self, total_t) -> float:
        return max(d for row in self.duty for d in row) / total_t


def wanted_naive(t, K, C, period):
    return set(range(K))                       # the same K columns, forever


def wanted_rotate(t, K, C, period):
    s = int(t // period) % C                   # window slides one column per period
    return {(s + i) % C for i in range(K)}


def run(policy_name, wanted_fn, sheet, F_target, K, hold_t, period, dt=0.02, settle=2.5):
    """settle: ignore the initial contraction transient when scoring force tracking."""
    n_force, n_lo, n_hi = 0, 1e9, -1e9
    sum_force, energy = 0.0, 0.0
    idle_peak = 0.0                             # hottest cell in an OFF column (parasitic)
    t = 0.0
    while t < hold_t - 1e-9:
        wanted = wanted_fn(t, K, sheet.cols, period)
        modes = sheet.col_modes(wanted)
        energy += sheet.rows * sum(sheet.cells[0][c].power(modes[c]) for c in range(sheet.cols)) * dt
        sheet.step(dt, modes)
        if t >= settle:                         # score tracking only after warmup
            f = sheet.force
            sum_force += f; n_force += 1
            n_lo, n_hi = min(n_lo, f), max(n_hi, f)
            # hottest cell whose column is NOT driven this step -> stray crosstalk heat
            off = [c for c in range(sheet.cols) if c not in wanted]
            if off:
                idle_peak = max(idle_peak,
                                max(sheet.cells[r][c].T for r in range(sheet.rows) for c in off))
        t += dt
    off_neighbor_peak = idle_peak
    return {
        "policy": policy_name,
        "mean_force": sum_force / n_force,
        "min_force": n_lo,
        "max_force": n_hi,
        "peak_temp": sheet.peak_temp,
        "max_duty": sheet.max_duty_frac(hold_t),
        "energy": energy,
        "idle_cell_peakT": off_neighbor_peak,
    }


if __name__ == "__main__":
    spec = SMACellSpec()                        # single 50 mm segments; cell = one wire
    R, C = 3, 8
    K = 4                                        # columns needed to hold the target
    F_target = K * spec.pull_force_n
    HOLD_T, PERIOD = 40.0, 5.0

    print(f"sheet {R}x{C} ({R*C} cells), col = {R} series (stroke {R*spec.single_stroke_mm:.0f} mm), "
          f"{C} parallel cols")
    print(f"target {F_target:.1f} N = {K} of {C} columns hot; hold {HOLD_T:.0f}s; "
          f"crosstalk g/h={0.4}")
    print(f"naive pins {K} columns; rotate sweeps the {K}-col window every {PERIOD:.0f}s\n")

    rows = []
    for name, fn in (("naive", wanted_naive), ("rotate", wanted_rotate)):
        rows.append(run(name, fn, Sheet(R, C, spec), F_target, K, HOLD_T, PERIOD))

    h = rows[0]
    print(f"  {'policy':>7} {'meanF':>7} {'minF':>7} {'peakT':>7} {'maxDuty':>8} "
          f"{'energy':>8} {'idleT':>7}")
    for r in rows:
        print(f"  {r['policy']:>7} {r['mean_force']:6.1f}N {r['min_force']:6.1f}N "
              f"{r['peak_temp']:6.0f}C {r['max_duty']*100:7.0f}% {r['energy']:7.0f}J "
              f"{r['idle_cell_peakT']:6.0f}C")

    print(f"\n  fatigue lever: max per-cell hot-duty {rows[0]['max_duty']*100:.0f}% (naive) "
          f"-> {rows[1]['max_duty']*100:.0f}% (rotate)")
    print(f"  cost: energy {rows[0]['energy']:.0f}J -> {rows[1]['energy']:.0f}J, "
          f"force ripple min {rows[1]['min_force']:.1f}N vs target {F_target:.1f}N")
    print(f"  crosstalk: idle corner cell reached {rows[0]['idle_cell_peakT']:.0f}C "
          f"(naive) — cells are NOT thermally independent")
