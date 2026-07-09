"""
First-order model of a DRAW-WIRE (string-pot / "tape-measure") position encoder — the
cheapest possible way to measure a long linear travel, and the sensor this CoreXY stage
closes its loop on.

The idea (user's): a spring-loaded reel wound with Dyneema, a rotary encoder on the reel
axis. The load pulls the wire out; the encoder reads reel angle; angle -> extension. Dirt
cheap ($2 magnetic encoder + a reel + a tape spring) and it measures the LOAD directly, so
— exactly like the capacitive rail scale — it sees belt stretch and backlash that a motor
encoder cannot.

THE ONE REAL GOTCHA — the reel radius grows as wire winds on:

    extension x   <->   turns unwound   <->   encoder angle θ
    but the pay-out radius r changes with how much wire is still on the reel, so θ(x) is
    NONLINEAR. Model the wire piling up at one axial station (a cheap reel, no level-wind):
        turns to store length L:   T(L) = (-r0 + sqrt(r0² + d·L/π)) / d      (r = r0 + T·d)
        encoder angle at ext x:    θ(x) = 2π·( T(C) − T(C − x) )             (C = total wire)
        pay-out radius:            r(x) = r0 + d·T(C − x)
    A naive constant-radius decode (x ≈ r_nom·θ) is then off by MILLIMETRES over the travel.
    Two cures: a LEVEL-WIND (spread one layer across the reel width — keeps r ~ constant), or
    just CALIBRATE the θ(x) curve. Calibrated, the residual is only encoder quantisation.

Accuracy budget (honest): the floor is the encoder LSB, Δx = r·(2π/N) ≈ 10 µm for a 12-bit
magnetic encoder on a Ø10 mm reel — position-dependent (finest when nearly paid out). On top
sit wire creep/stretch under the spring tension (Dyneema is stiff + light, so small and mostly
a re-homable offset) and cosine error if the wire isn't along the axis. So ~tens of µm from
~$3 of parts over a long range — not the cap scale's ~1 µm, but the point of the experiment.

    ../.venv/bin/python corexy/drawwire.py            # report + out/drawwire.png
"""

from dataclasses import dataclass
from math import pi, sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


@dataclass
class DrawWireEncoder:
    travel_mm: float = 300.0          # measured travel
    dead_mm: float = 30.0             # dead wire always on the reel (never pays out)
    reel_hub_dia_mm: float = 10.0     # bare hub diameter (2·r0)
    wire_dia_mm: float = 0.5          # Dyneema diameter
    reel_width_mm: float = 10.0       # axial reel width (level-wind spreads across this)
    level_wind: bool = False          # True: one layer spread across width (r ~ constant)
    encoder_counts: int = 4096        # counts/rev (12-bit magnetic absolute; count turns)
    spring_force_N: float = 5.0       # constant-force return spring tension
    wire_modulus_GPa: float = 110.0   # Dyneema SK99 ~ 100-120 GPa
    creep_ppm: float = 200.0          # slow length creep under tension [ppm of paid-out len]

    # ---- reel geometry --------------------------------------------------------
    @property
    def r0(self) -> float:                       # hub radius [mm]
        return self.reel_hub_dia_mm / 2

    @property
    def d(self) -> float:
        return self.wire_dia_mm

    @property
    def C(self) -> float:                        # total wire wound at zero extension [mm]
        return self.travel_mm + self.dead_mm

    @property
    def _b(self) -> float:
        """Radius-growth coefficient: wound wire of length L fills an annulus over the reel
        width W, so R² = r0² + b·L with b = d²/(4W)  (multi-layer packing)."""
        return self.d**2 / (4 * self.reel_width_mm)

    def _radius(self, L):
        """Pay-out radius [mm] when wound length is L [mm]."""
        if self.level_wind:                      # one layer spread across width -> ~constant
            return self.r0 + 0.5 * self.d + 0 * np.asarray(L, dtype=float)
        return np.sqrt(self.r0**2 + self._b * np.asarray(L, dtype=float))

    def payout_radius(self, x):
        """Instantaneous pay-out radius at extension x [mm]."""
        return self._radius(self.C - np.asarray(x, dtype=float))

    def angle(self, x):
        """Encoder angle [rad] at extension x [mm].  θ = ∫dx/R has a closed form:
        θ(x) = (2/b)·(R(C) − R(C−x))  for the annulus reel; x/R for a level-wind."""
        x = np.asarray(x, dtype=float)
        if self.level_wind:
            return x / (self.r0 + 0.5 * self.d)
        return (2 / self._b) * (self._radius(self.C) - self._radius(self.C - x))

    @property
    def r_nom(self) -> float:
        """Best single-radius guess (mid-travel pay-out radius) for the naive decode."""
        return float(self.payout_radius(self.travel_mm / 2))

    # ---- resolution & decode --------------------------------------------------
    def resolution(self, x):
        """Position resolution [mm] per encoder count at extension x."""
        return self.payout_radius(x) * (2 * pi / self.encoder_counts)

    def decode_naive(self, x):
        """What you read if you assume a CONSTANT radius r_nom (uncalibrated)."""
        return self.r_nom * self.angle(x)

    @property
    def stretch_strain(self) -> float:
        """Elastic strain of the wire under the constant spring tension (a fixed gain,
        so it CALIBRATES OUT — not a per-reading error)."""
        return self.spring_force_N / (self.wire_modulus_GPa * 1e3 * (pi / 4 * self.d**2))

    def decode(self, x, rng=None):
        """CALIBRATED decode: quantise the true angle, invert the modelled θ(x). Returns the
        measured extension [mm]. The floor is encoder quantisation; the wire's constant-tension
        stretch is a calibrated gain and creep is a slow re-homable drift (see report)."""
        rng = rng or np.random.default_rng(0)
        x = np.atleast_1d(np.asarray(x, dtype=float))
        theta = self.angle(x)
        lsb = 2 * pi / self.encoder_counts
        theta_q = np.round(theta / lsb) * lsb + rng.normal(0, 0.3 * lsb, size=x.shape)
        grid = np.linspace(0, self.travel_mm, 20001)
        return np.interp(theta_q, self.angle(grid), grid)

    @property
    def noise_rms(self) -> float:
        """Per-reading position noise [mm] the servo actually sees (quantisation-limited)."""
        rng = np.random.default_rng(7)
        xt = np.linspace(0, self.travel_mm, 3000)
        return float(np.std(self.decode(xt, rng) - xt))

    # ---- validation -----------------------------------------------------------
    @property
    def max_nonlinearity(self) -> float:
        """Worst naive-decode error after removing the best-fit line [mm] — what calibration
        (or a level-wind) must handle."""
        x = np.linspace(0, self.travel_mm, 400)
        naive = self.decode_naive(x)
        a, b = np.polyfit(x, naive, 1)
        return float(np.max(np.abs(naive - (a * x + b))))

    def checks(self):
        c = []
        c.append(("travel spans at least one turn",
                  self.angle(self.travel_mm) > 2 * pi,
                  f"{self.angle(self.travel_mm)/(2*pi):.1f} turns over travel"))
        r_max = float(self.payout_radius(0.0))
        c.append(("reel stays compact",
                  r_max < 4 * self.r0,
                  f"R_max {r_max:.1f} mm vs hub r0 {self.r0:.1f} mm"))
        c.append(("resolution useful (< 50 µm across travel)",
                  self.resolution(0.0) < 0.05,
                  f"{self.resolution(0.0)*1e3:.1f} µm (worst, retracted)"))
        c.append(("wire tension well below Dyneema break",
                  self.spring_force_N < 0.5 * 3.0e3 * (pi / 4 * (self.d * 1e-3)**2 * 1e6),
                  f"{self.spring_force_N:.1f} N vs ~{3.0e3*(pi/4*(self.d*1e-3)**2*1e6):.0f} N break"))
        c.append(("nonlinearity is real but bounded (calibratable)",
                  self.max_nonlinearity < 0.5 * self.travel_mm,
                  f"{self.max_nonlinearity:.2f} mm naive error (calibrate it out)"))
        return c

    @property
    def is_valid(self) -> bool:
        return all(ok for _, ok, _ in self.checks())


def report(p: DrawWireEncoder | None = None):
    p = p or DrawWireEncoder()
    print("\n" + "=" * 66)
    print("  DRAW-WIRE (string-pot) ENCODER  —  the tape-measure sensor")
    print("=" * 66)
    print(f"  travel                 {p.travel_mm:8.0f} mm")
    print(f"  reel hub / wire / width {p.reel_hub_dia_mm:.0f} / {p.wire_dia_mm} / "
          f"{p.reel_width_mm:.0f} mm   level-wind: {p.level_wind}")
    print(f"  pay-out radius         {p.payout_radius(p.travel_mm):.2f} .. "
          f"{p.payout_radius(0.0):.2f} mm   (grows as it winds in)")
    print(f"  encoder                {p.encoder_counts} counts/rev "
          f"({np.log2(p.encoder_counts):.0f}-bit)  → {p.angle(p.travel_mm)/(2*pi):.1f} turns")
    print("-" * 66)
    print(f"  resolution             {p.resolution(p.travel_mm)*1e3:.1f} µm (paid out) .. "
          f"{p.resolution(0.0)*1e3:.1f} µm (retracted)")
    print(f"  naive nonlinearity     {p.max_nonlinearity:.2f} mm   "
          f"<- if you assume a fixed radius (calibrate or level-wind it out)")
    rng = np.random.default_rng(1)
    xt = np.linspace(0, p.travel_mm, 2000)
    err = p.decode(xt, rng) - xt
    print(f"  calibrated decode      max |err| {np.max(np.abs(err))*1e3:.1f} µm, "
          f"RMS {np.std(err)*1e3:.1f} µm  <- the per-reading floor the servo sees")
    print("  budget beyond that (both removable, NOT per-reading noise):")
    print(f"    · tension stretch    {p.stretch_strain*p.travel_mm*1e3:5.0f} µm at full ext "
          f"— constant gain, CALIBRATED out")
    print(f"    · wire creep         {p.creep_ppm*1e-6*p.travel_mm*1e3:5.0f} µm at full ext "
          f"— slow drift, RE-HOMED out")
    print("-" * 66)
    for name, ok, detail in p.checks():
        print(f"    [{'ok ' if ok else 'XX '}] {name:<44} {detail}")
    print(f"  -> {'VALID' if p.is_valid else 'INVALID'}\n" + "=" * 66 + "\n")
    return p


def render(p: DrawWireEncoder | None = None):
    p = p or DrawWireEncoder()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.linspace(0, p.travel_mm, 600)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # (a) pay-out radius + resolution vs extension
    ax[0, 0].plot(x, p.payout_radius(x), color="C0", lw=1.8)
    ax[0, 0].set_title("(a) pay-out radius grows as wire winds in")
    ax[0, 0].set_xlabel("extension  [mm]"); ax[0, 0].set_ylabel("radius  [mm]", color="C0")
    ax[0, 0].tick_params(axis="y", labelcolor="C0"); ax[0, 0].grid(alpha=0.3)
    axr = ax[0, 0].twinx()
    axr.plot(x, p.resolution(x) * 1e3, color="C1", lw=1.5)
    axr.set_ylabel("resolution  [µm/count]", color="C1"); axr.tick_params(axis="y", labelcolor="C1")

    # (b) θ(x): actual (nonlinear) vs best-fit line
    th = p.angle(x)
    a, b = np.polyfit(x, th, 1)
    ax[0, 1].plot(x, th, color="C0", lw=1.8, label="actual θ(x)")
    ax[0, 1].plot(x, a * x + b, "k--", lw=0.9, label="linear fit")
    ax[0, 1].set_title("(b) angle↔extension is NONLINEAR (radius growth)")
    ax[0, 1].set_xlabel("extension  [mm]"); ax[0, 1].set_ylabel("encoder angle  [rad]")
    ax[0, 1].legend(fontsize=8, loc="upper left"); ax[0, 1].grid(alpha=0.3)

    # (c) decode error: naive constant-radius vs calibrated
    an, bn = np.polyfit(x, p.decode_naive(x), 1)
    ax[1, 0].plot(x, (p.decode_naive(x) - (an * x + bn)) * 1e0, color="C3", lw=1.5,
                  label="naive (fixed radius)")
    xe = np.linspace(0, p.travel_mm, 1500)
    cal_err = (p.decode(xe, np.random.default_rng(4)) - xe) * 1e3
    axc = ax[1, 0].twinx()
    axc.plot(xe, cal_err, color="C2", lw=0.6, alpha=0.7)
    ax[1, 0].set_title("(c) decode error: naive [mm] vs calibrated [µm]")
    ax[1, 0].set_xlabel("extension  [mm]")
    ax[1, 0].set_ylabel("naive error  [mm]", color="C3"); ax[1, 0].tick_params(axis="y", labelcolor="C3")
    axc.set_ylabel("calibrated error  [µm]", color="C2"); axc.tick_params(axis="y", labelcolor="C2")
    ax[1, 0].legend(fontsize=8, loc="upper left"); ax[1, 0].grid(alpha=0.3)

    # (d) design lever: nonlinearity vs reel width, and level-wind fix
    widths = np.linspace(5, 40, 40)
    nl = [DrawWireEncoder(**{**p.__dict__, "reel_width_mm": w}).max_nonlinearity for w in widths]
    lw_nl = DrawWireEncoder(**{**p.__dict__, "level_wind": True}).max_nonlinearity
    ax[1, 1].plot(widths, nl, color="C4", lw=1.8, label="pile-up reel")
    ax[1, 1].axhline(lw_nl, color="C2", ls="--", lw=1.2, label="level-wind (any width)")
    ax[1, 1].scatter([p.reel_width_mm], [p.max_nonlinearity], color="C4", zorder=5)
    ax[1, 1].set_title("(d) wider reel / level-wind → less nonlinearity")
    ax[1, 1].set_xlabel("reel width  [mm]"); ax[1, 1].set_ylabel("naive nonlinearity  [mm]")
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)

    fig.suptitle(
        f"Draw-wire encoder — {p.resolution(p.travel_mm)*1e3:.0f}–{p.resolution(0)*1e3:.0f} µm "
        f"over {p.travel_mm:.0f} mm from a reel + rotary encoder (calibrate the radius growth)",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "drawwire.png", dpi=120)
    print(f"  wrote {OUT/'drawwire.png'}")


if __name__ == "__main__":
    p = report()
    render(p)
