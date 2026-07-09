"""
First-order model of the CAPACITIVE VERNIER LINEAR ENCODER — the digital-caliper
scale that turns a garbage TT-gearmotor belt axis into a precision servo.

The thesis of this whole folder: precision belongs to the *sensor*, not the actuator.
A TT gearmotor has ~1/4 mm of gearbox backlash and no positional repeatability, but if
you measure the *carriage* directly with a ~1 µm scale and close the loop, all that
mechanical slop is inside the loop and gets cancelled. This file models the scale;
servo.py proves the cancellation; rail.py draws the hardware.

HOW A CAPACITIVE CALIPER WORKS (what we're reverse-engineering):
Two PCBs face each other across a thin air gap g (~0.2 mm). The moving SLIDER carries
transmitter fingers driven by an n-phase excitation and a patterned receiver comb; the
fixed SCALE carries coupling electrodes at a fine pitch P. As the slider translates by
x, the transmitter->receiver coupling capacitance is modulated sinusoidally with spatial
period P, so the demodulated signal phase is

    φ(x) = 2π·x / P            (mod 2π)  — position WITHIN one pitch, wraps every P.

Quadrature demodulation (I = C_sig·cosφ, Q = C_sig·sinφ) recovers φ to a noise floor set
by the front-end capacitance resolution σ_C against the modulated signal amplitude C_sig:

    σ_φ ≈ σ_C / C_sig          σ_x(fine) = (P/2π)·σ_φ

C_sig is real parallel-plate coupling, C_sig = κ·ε0·A_active/g, so resolution follows
directly from electrode area, air gap, and the ASIC noise floor — not a fudge factor.

THE VERNIER (why "like a caliper" and how it becomes ABSOLUTE):
One fine track wraps every P — it is incremental. Run TWO tracks at slightly different
pitches P1, P2. Each gives a phase; their DIFFERENCE advances one full turn only over the
beat (nonius) length

    L_beat = P1·P2 / |P1 − P2|

so Δφ = φ1 − φ2 is a single monotone ramp 0→2π across L_beat — a coarse ABSOLUTE channel.
Round the coarse estimate to the nearest fine period, then read the fine phase inside it:

    k = round((x_coarse − x_fine)/P1),   x_abs = k·P1 + x_fine

This is the mechanical vernier/nonius principle done in copper: the beat names the period,
the fine phase places you within it. Absolute, valveless, sub-micron, ~free on a PCB.
Unambiguous ONLY while the coarse channel can resolve one fine period, i.e. its noise
σ_x(coarse) ≪ P1/2 — the model checks that margin.

Everything here is analytical first-order (ideal sinusoidal coupling, Gaussian phase noise,
parallel-plate C with a lumped modulation depth κ). Trustworthy: the resolution scaling
with gap/area/noise, the absolute range = L_beat, and the period-stitch margin. NOT modelled:
fringing exactness, tilt/runout of the gap, temperature drift of ε and geometry.

    ../.venv/bin/python linear-rail-servo/encoder.py            # report + out/encoder.png
"""

from dataclasses import dataclass
from math import pi
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

EPS0 = 8.8541878128e-12   # vacuum permittivity [F/m]


@dataclass
class CapVernierEncoder:
    # --- travel & the two vernier tracks ---------------------------------------
    travel_mm: float = 150.0         # usable carriage travel (must be <= L_beat)
    pitch1_mm: float = 5.08          # fine track A pitch (0.2", the caliper standard)
    pitch2_mm: float = 5.00          # fine track B pitch — the vernier partner (P1 != P2)

    # --- electrode geometry (sets the coupling capacitance -> the noise floor) --
    gap_mm: float = 0.20             # electrode air gap g (slider PCB to scale PCB)
    track_width_mm: float = 8.0      # transverse electrode width
    recv_len_mm: float = 30.0        # receiver comb length along travel (active footprint)
    n_phases: int = 8                # excitation phases (caliper ASICs use 8)
    mod_depth: float = 0.5           # κ: position-modulated fraction of the coupling C
    eps_r_gap: float = 1.0           # air in the gap

    # --- front-end electronics -------------------------------------------------
    excite_vpp: float = 3.3          # excitation amplitude (sets absolute signal, not res.)
    cap_noise_fF: float = 5.0        # RMS capacitance resolution of the read-out [fF]

    # ---- geometry (SI) --------------------------------------------------------
    @property
    def p1(self) -> float:                 # fine pitch A [m]
        return self.pitch1_mm * 1e-3

    @property
    def p2(self) -> float:                 # fine pitch B [m]
        return self.pitch2_mm * 1e-3

    @property
    def travel(self) -> float:             # [m]
        return self.travel_mm * 1e-3

    @property
    def gap(self) -> float:                # [m]
        return self.gap_mm * 1e-3

    @property
    def beat_len(self) -> float:
        """Vernier (nonius) beat length = absolute unambiguous range [m]."""
        return self.p1 * self.p2 / abs(self.p1 - self.p2)

    @property
    def signed_beat(self) -> float:
        """Beat with sign: P1·P2/(P2−P1). Negative when P1 > P2 (beat runs backwards).
        Using the signed value keeps the coarse decode increasing with x for either order."""
        return self.p1 * self.p2 / (self.p2 - self.p1)

    # ---- capacitive signal & phase noise --------------------------------------
    @property
    def active_area(self) -> float:        # coherent modulated footprint [m^2]
        return (self.track_width_mm * 1e-3) * (self.recv_len_mm * 1e-3)

    @property
    def coupling_cap(self) -> float:       # full plate coupling C over the footprint [F]
        return self.eps_r_gap * EPS0 * self.active_area / self.gap

    @property
    def signal_cap(self) -> float:         # amplitude of the position-modulated C [F]
        return self.mod_depth * self.coupling_cap

    @property
    def cap_noise(self) -> float:          # front-end capacitance noise [F]
        return self.cap_noise_fF * 1e-15

    @property
    def snr(self) -> float:                # modulated-signal SNR
        return self.signal_cap / self.cap_noise

    @property
    def phase_noise(self) -> float:        # σ_φ per channel [rad]
        return 1.0 / self.snr

    @property
    def res_fine(self) -> float:           # within-period position noise (fine) [m]
        return self.p1 / (2 * pi) * self.phase_noise

    @property
    def res_coarse(self) -> float:
        """Coarse (vernier) position noise [m]. The phase DIFFERENCE of two independent
        channels carries √2 the phase noise and is stretched over the beat length."""
        return self.beat_len / (2 * pi) * (np.sqrt(2) * self.phase_noise)

    @property
    def stitch_margin(self) -> float:
        """How comfortably the coarse channel resolves one fine period: (P1/2)/σ_coarse.
        > ~4 means period identification is reliable; < 1 means it slips."""
        return (self.p1 / 2) / self.res_coarse

    @property
    def eff_bits(self) -> float:           # effective absolute resolution over travel
        return np.log2(self.travel / self.res_fine)

    @property
    def origin(self) -> float:
        """Scale-registration offset [m]: where mechanical x=0 sits along the beat.
        We centre the travel inside the beat so the coarse WRAP SEAM (at beat-coord 0)
        lands in the unused margin, never inside [0, travel]. Free choice of where you
        glue the scale down — but it's what makes the two-track vernier truly absolute."""
        return max(0.0, (self.beat_len - self.travel) / 2)

    # ---- the decode (the actual algorithm, with noise) ------------------------
    def _phases(self, x, rng):
        """Measured wrapped phases (φ1, φ2) at scale positions x [m], with I/Q noise."""
        phi1t = 2 * pi * x / self.p1
        phi2t = 2 * pi * x / self.p2
        sC = self.phase_noise            # per-channel phase-equivalent noise
        # add Gaussian noise in quadrature, recover phase via atan2 (proper wrapping)
        n = lambda: rng.normal(0.0, sC, size=np.shape(x))
        phi1 = np.arctan2(np.sin(phi1t) + n(), np.cos(phi1t) + n())
        phi2 = np.arctan2(np.sin(phi2t) + n(), np.cos(phi2t) + n())
        return np.mod(phi1, 2 * pi), np.mod(phi2, 2 * pi)

    def decode_parts(self, x, rng=None):
        """True mechanical position x [m] -> (coarse, fine, absolute) all in mechanical
        coords. Works in beat-coords (x + origin) internally so the seam stays clear."""
        rng = rng or np.random.default_rng(0)
        x = np.atleast_1d(np.asarray(x, dtype=float))
        xb = x + self.origin                        # position along the scale (beat-coord)
        phi1, phi2 = self._phases(xb, rng)
        dphi = np.mod(phi1 - phi2, 2 * pi)          # vernier beat phase [0, 2π)
        # signed beat keeps the coarse position increasing with x regardless of P1<>P2
        x_coarse = np.mod(self.signed_beat * dphi / (2 * pi), self.beat_len)
        x_fine = self.p1 * phi1 / (2 * pi)          # fine within one P1 period [0,P1)
        k = np.round((x_coarse - x_fine) / self.p1)  # which fine period
        x_abs = k * self.p1 + x_fine
        return x_coarse - self.origin, x_fine, x_abs - self.origin

    def decode(self, x, rng=None):
        """True position x [m] -> measured ABSOLUTE position [m] via vernier stitch."""
        return self.decode_parts(x, rng)[2]

    # ---- validation -----------------------------------------------------------
    def checks(self):
        c = []
        c.append(("two distinct pitches (vernier exists)",
                  self.pitch1_mm != self.pitch2_mm,
                  f"P1={self.pitch1_mm}, P2={self.pitch2_mm} mm"))
        c.append(("travel within absolute range (L_beat)",
                  self.travel <= self.beat_len,
                  f"travel {self.travel_mm:.0f} <= beat {self.beat_len*1e3:.0f} mm"))
        c.append(("coarse channel resolves one fine period",
                  self.stitch_margin > 4.0,
                  f"margin (P1/2)/σ_coarse = {self.stitch_margin:.1f}x"))
        c.append(("signal SNR adequate for interpolation",
                  self.snr > 100.0,
                  f"SNR = {self.snr:.0f} (C_sig {self.signal_cap*1e12:.2f}pF / {self.cap_noise_fF:g}fF)"))
        c.append(("fine resolution below one pitch",
                  self.res_fine < self.p1,
                  f"σ_fine {self.res_fine*1e6:.2f} µm << P1 {self.pitch1_mm} mm"))
        c.append(("quadrature-capable phase count",
                  self.n_phases >= 4,
                  f"{self.n_phases} phases"))
        c.append(("air gap manufacturable",
                  0.05e-3 <= self.gap <= 0.5e-3,
                  f"{self.gap_mm} mm"))
        c.append(("effective resolution useful",
                  self.eff_bits > 14.0,
                  f"{self.eff_bits:.1f} bits over {self.travel_mm:.0f} mm"))
        return c

    @property
    def is_valid(self) -> bool:
        return all(ok for _, ok, _ in self.checks())


def report(p: CapVernierEncoder | None = None):
    p = p or CapVernierEncoder()
    print("\n" + "=" * 68)
    print("  CAPACITIVE VERNIER LINEAR ENCODER  (digital-caliper scale)")
    print("=" * 68)
    print(f"  travel                 {p.travel_mm:8.1f} mm")
    print(f"  fine pitches  P1 / P2  {p.pitch1_mm:8.2f} / {p.pitch2_mm:.2f} mm  (Δ {abs(p.pitch1_mm-p.pitch2_mm):.2f})")
    print(f"  vernier beat  L_beat   {p.beat_len*1e3:8.1f} mm   <- absolute range")
    print(f"  air gap  g             {p.gap_mm:8.2f} mm")
    print(f"  active footprint       {p.track_width_mm:.0f} × {p.recv_len_mm:.0f} mm  ({p.active_area*1e6:.0f} mm²)")
    print("-" * 68)
    print(f"  coupling C (full)      {p.coupling_cap*1e12:8.2f} pF")
    print(f"  modulated C_sig (κ={p.mod_depth:g})  {p.signal_cap*1e12:6.2f} pF")
    print(f"  read-out noise         {p.cap_noise_fF:8.2f} fF   ->  SNR {p.snr:.0f}")
    print(f"  phase noise  σ_φ       {p.phase_noise*1e3:8.3f} mrad")
    print("-" * 68)
    print(f"  FINE resolution σ_x    {p.res_fine*1e6:8.2f} µm   (within a pitch)")
    print(f"  COARSE (vernier) σ_x   {p.res_coarse*1e6:8.1f} µm   (names the period)")
    print(f"  period-stitch margin   {p.stitch_margin:8.1f} ×    (want > 4)")
    print(f"  effective resolution   {p.eff_bits:8.1f} bits over {p.travel_mm:.0f} mm")
    print("-" * 68)

    # end-to-end decode sweep with noise
    rng = np.random.default_rng(1)
    xt = np.linspace(0, p.travel * 0.999, 4001)
    xd = p.decode(xt, rng)
    err = xd - xt
    slips = int(np.sum(np.abs(err) > p.p1 / 2))       # period-identification failures
    print(f"  decode sweep (4001 pts): max |err| {np.max(np.abs(err))*1e6:6.1f} µm, "
          f"RMS {np.std(err)*1e6:.2f} µm, period slips {slips}")
    print("-" * 68)
    print("  checks:")
    for name, ok, detail in p.checks():
        print(f"    [{'ok ' if ok else 'XX '}] {name:<42} {detail}")
    print(f"  -> {'VALID' if p.is_valid else 'INVALID'} configuration")
    print("=" * 68 + "\n")
    return p


def render(p: CapVernierEncoder | None = None):
    p = p or CapVernierEncoder()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(2)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # (a) the two fine channel phases over a few pitches — slightly different periods
    xw = np.linspace(0, 3 * p.pitch1_mm, 1500) * 1e-3
    phi1, phi2 = p._phases(xw, rng)
    ax[0, 0].plot(xw * 1e3, phi1 / (2 * pi), lw=1.4, label=f"track A  P1={p.pitch1_mm} mm")
    ax[0, 0].plot(xw * 1e3, phi2 / (2 * pi), lw=1.4, label=f"track B  P2={p.pitch2_mm} mm")
    ax[0, 0].set_title("(a) fine channels wrap — each is incremental")
    ax[0, 0].set_xlabel("position  [mm]"); ax[0, 0].set_ylabel("phase / 2π")
    ax[0, 0].legend(fontsize=8, loc="upper right"); ax[0, 0].grid(alpha=0.3)

    # (b) the vernier beat across the whole scale — one clean absolute ramp, and where
    #     the travel is registered so the wrap seam stays out of range
    xb = np.linspace(0, p.beat_len * 0.999, 4000)          # scale (beat) coordinate
    p1f, p2f = p._phases(xb, np.random.default_rng(3))
    dphi = np.mod(p1f - p2f, 2 * pi)
    x_coarse = np.mod(p.signed_beat * dphi / (2 * pi), p.beat_len)
    ax[0, 1].plot(xb * 1e3, x_coarse * 1e3, lw=0.8, color="C3", label="coarse (vernier alone)")
    ax[0, 1].axvspan(p.origin * 1e3, (p.origin + p.travel) * 1e3, color="C2", alpha=0.14,
                     label="travel (seam parked outside)")
    ax[0, 1].set_title(f"(b) vernier difference → absolute over L_beat = {p.beat_len*1e3:.0f} mm")
    ax[0, 1].set_xlabel("scale position  [mm]"); ax[0, 1].set_ylabel("coarse decoded  [mm]")
    ax[0, 1].legend(fontsize=8, loc="upper left"); ax[0, 1].grid(alpha=0.3)

    # (c) decoded absolute vs true, with the residual error
    xt = np.linspace(0, p.travel * 0.999, 3000)
    xd = p.decode(xt, np.random.default_rng(4))
    err = (xd - xt) * 1e6
    ax[1, 0].plot(xt * 1e3, xd * 1e3, lw=1.0, color="C0")
    ax[1, 0].plot(xt * 1e3, xt * 1e3, "k--", lw=0.7, alpha=0.6, label="ideal y=x")
    ax[1, 0].set_title("(c) decoded ABSOLUTE position vs true")
    ax[1, 0].set_xlabel("true position  [mm]"); ax[1, 0].set_ylabel("decoded  [mm]")
    axr = ax[1, 0].twinx()
    axr.plot(xt * 1e3, err, color="C1", lw=0.5, alpha=0.5)
    axr.set_ylabel("residual  [µm]", color="C1")
    axr.tick_params(axis="y", labelcolor="C1")
    lim = max(5.0, 4 * np.std(err)); axr.set_ylim(-lim, lim)
    ax[1, 0].legend(fontsize=8, loc="upper left"); ax[1, 0].grid(alpha=0.3)

    # (d) the two design levers, each on its own honest axis:
    #     fine resolution set by the air gap; absolute range set by the pitch mismatch ΔP.
    gaps = np.linspace(0.08, 0.45, 60)
    resg = [CapVernierEncoder(**{**p.__dict__, "gap_mm": g}).res_fine * 1e6 for g in gaps]
    ax[1, 1].plot(gaps, resg, color="C4", lw=1.8)
    ax[1, 1].scatter([p.gap_mm], [p.res_fine * 1e6], color="C4", zorder=5)
    ax[1, 1].annotate(f"  {p.res_fine*1e6:.1f} µm @ {p.gap_mm} mm",
                      (p.gap_mm, p.res_fine * 1e6), fontsize=8, color="C4")
    ax[1, 1].set_title("(d) two levers: gap → resolution, ΔP → absolute range")
    ax[1, 1].set_xlabel("air gap  g  [mm]  (∝ resolution)")
    ax[1, 1].set_ylabel("fine σ_x  [µm]", color="C4")
    ax[1, 1].tick_params(axis="y", labelcolor="C4"); ax[1, 1].grid(alpha=0.3)
    axt = ax[1, 1].twiny(); axb = axt.twinx()          # independent top-x / right-y pair
    dps = np.linspace(0.02, 0.30, 60)
    beats = [p.pitch1_mm * (p.pitch1_mm - d) / d for d in dps]   # L_beat(ΔP) [mm]
    axb.plot(dps, beats, color="C5", lw=1.8)
    axb.scatter([abs(p.pitch1_mm - p.pitch2_mm)], [p.beat_len * 1e3], color="C5", zorder=5)
    axb.axhline(p.travel_mm, color="C2", ls="--", lw=0.8)
    axt.set_xlabel("pitch mismatch  ΔP  [mm]  (∝ 1/range)", color="C5")
    axt.tick_params(axis="x", labelcolor="C5")
    axb.set_ylabel("L_beat  [mm]", color="C5"); axb.tick_params(axis="y", labelcolor="C5")

    fig.suptitle(
        f"Capacitive vernier scale — {p.res_fine*1e6:.1f} µm fine, "
        f"absolute over {p.beat_len*1e3:.0f} mm, from two caliper tracks",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "encoder.png", dpi=120)
    print(f"  wrote {OUT/'encoder.png'}")


if __name__ == "__main__":
    p = report()
    render(p)
