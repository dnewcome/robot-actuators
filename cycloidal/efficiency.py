"""
Layer B (v0): a first-order efficiency model for the cycloidal drive.

It answers the question "can rolling elements make this more efficient, and what does
it cost in size?" by (1) predicting η per contact strategy and (2) sweeping real needle
bearings for fit.

HONESTY: the absolute η values are estimates with explicitly tunable coefficients
(L_BASE / S_RING / S_OUT). One torque-meter point calibrates them. The *relative ranking*
(printed < steel < sleeve < needle) is robust because it is driven by the friction-
coefficient ratios, which are well established. This is the analytical half of Layer B;
the experimental half is the single measured point that pins down the constants.

Run:  python cycloidal/efficiency.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drive import Params  # noqa: E402
from math import pi

# --------------------------------------------------------------------------- #
# Friction coefficients per contact type (dimensionless). Ranges in comments.
# --------------------------------------------------------------------------- #
CONTACT_FRICTION = {
    "printed":     0.20,   # lobe SLIDES on a fixed printed (resin/plastic) pin — tacky, high (0.15-0.3)
    "steel_dry":   0.12,   # lobe SLIDES on a fixed dry steel pin (0.10-0.15)
    "steel_lubed": 0.09,   # ... lightly lubricated (0.07-0.11)
    "sleeve":      0.03,   # lobe ROLLS on a free sleeve; sleeve slides on its post (plain) — effective (0.02-0.05)
    "needle":      0.006,  # lobe rolls on a needle-bearing roller — rolling throughout (0.004-0.008)
}

# Model coefficients — CALIBRATION-PENDING (set so the anchors below are sensible).
#   all-printed   -> ~0.67     all-steel(dry) -> ~0.79
#   sleeve+steel  -> ~0.87 (the current default)   all-needle -> ~0.92
L_BASE = 0.06   # eccentric ball bearing (heavily loaded) + churn — ~independent of pin choice
S_RING = 1.0    # ring-pin mesh carries the full reduction load -> dominant loss term
S_OUT  = 0.45   # output-pin / disc-hole contact: lighter load, shorter slip path


def predict(ring_contact: str, out_contact: str) -> float:
    """Predicted ASYMPTOTIC (high-load) efficiency η∞ for a contact strategy.
    This is the ceiling the load-dependent curve approaches; see eta_at_load()."""
    mu_r = CONTACT_FRICTION[ring_contact]
    mu_o = CONTACT_FRICTION[out_contact]
    eta = 1.0 - (L_BASE + S_RING * mu_r + S_OUT * mu_o)
    return max(0.0, min(1.0, eta))


# --------------------------------------------------------------------------- #
# Load-dependent efficiency (torque-based model)
# --------------------------------------------------------------------------- #
# A roughly constant no-load drag must be overcome before any useful output torque
# appears, so η rises from 0 (no load) toward η∞ (high load). This is exactly the
# behavior MuJoCo realizes when the gear carries η∞ and the joint carries a Coulomb
# frictionloss = drag: T_out = η∞·N·τ_motor − drag, hence the form below.
DRAG_TORQUE_IN = 0.002   # N·m, input-referred no-load drag (bearings + pin preload + churn).
                         # Output-referred drag = this × ratio. CALIBRATION-PENDING: the
                         # torque meter's no-load reading sets it directly (I0 = drag/Kt).


def drag_out(p: Params, drag_in: float = DRAG_TORQUE_IN) -> float:
    """No-load drag referred to the output shaft."""
    return drag_in * p.ratio


def eta_at_load(T_out: float, eta_inf: float, drag_out_nm: float) -> float:
    """Efficiency at a given output torque — the curve the sim physically realizes.
        η(T) = η∞ · T / (T + drag_out)
    Rises from 0 at no load to η∞ at high load."""
    if T_out <= 0:
        return 0.0
    return eta_inf * T_out / (T_out + drag_out_nm)


def params_to_contacts(p: Params):
    """Map the CAD's pin_mode / out_mode onto contact types."""
    if p.pin_mode == "fixed":
        ring = "steel_dry"                         # press-fit steel dowel, lobe slides
    elif p.pin_core_dia < p.pin_dia:
        ring = "needle"                            # thin core + sleeve = room for a needle bearing
    else:
        ring = "sleeve"                            # free-spinning bare dowel (plain rolling)
    out = {"printed": "printed", "steel": "steel_dry", "bushing": "sleeve"}[p.out_mode]
    return ring, out


def predict_params(p: Params) -> float:
    return predict(*params_to_contacts(p))


# --------------------------------------------------------------------------- #
# Real needle bearings (and a build-your-own loose-needle option), bore/OD/width mm.
# Treating OD as the roller (= ring pin) OD that the lobe contacts.
# --------------------------------------------------------------------------- #
NEEDLE_BEARINGS = [
    # name                                bore  OD  width
    ("loose Ø0.5 needles on Ø1.5 post",   1.5,  2.5,  4),   # full-complement, build-your-own
    ("loose Ø1.0 needles on Ø2.0 post",   2.0,  4.0,  4),
    ("HK0408 drawn cup",                  4.0,  8.0,  8),
    ("HK0509 drawn cup",                  5.0,  9.0,  9),
]


def min_drive_for_pin_od(p: Params, roller_od: float, pin_gap: float = 1.0):
    """Smallest pin circle / housing OD that fits N rollers of `roller_od` without collision."""
    # pin pitch (π·D_pc/N) must exceed roller_od + a gap
    d_pc_min = p.n_pins * (roller_od + pin_gap) / pi
    housing_od = d_pc_min + roller_od + 2 * p.housing_wall
    return d_pc_min, housing_od


def loss_map():
    print("=== Where the losses are (cycloidal) ===")
    rows = [
        ("ring pin / lobe", "main reduction torque", "dominant", "fixed pin SLIDES -> roller/needle = ROLLING"),
        ("output pin / hole", "output torque", "secondary", "printed slide -> steel -> bushing/roller"),
        ("eccentric bearing", "reacts the whole mesh force", "fixed ~L_BASE", "already a ball bearing; heavily loaded"),
        ("churning / windage", "—", "tiny (dry, small)", "negligible"),
    ]
    for what, carries, share, lever in rows:
        print(f"  {what:18s} carries {carries:26s} loss: {share:14s} {lever}")
    print()


def config_table():
    print("=== Predicted efficiency by contact strategy (calibration-pending) ===")
    configs = [
        ("all printed (worst)",        "printed",     "printed"),
        ("steel pins, sliding",        "steel_dry",   "steel_dry"),
        ("DEFAULT: rolling pins + steel out", "sleeve", "steel_dry"),
        ("sleeves both",               "sleeve",      "sleeve"),
        ("needle ring + steel out",    "needle",      "steel_dry"),
        ("needle bearings everywhere", "needle",      "needle"),
    ]
    print(f"  {'config':38s} {'ring':10s} {'output':10s}  η")
    for name, r, o in configs:
        print(f"  {name:38s} {r:10s} {o:10s}  {predict(r, o)*100:4.0f}%")
    print()


def bearing_sweep(p: Params):
    pitch = pi * p.pin_circle_dia / p.n_pins
    print(f"=== Needle-bearing fit sweep (current: Ø{p.pin_dia} pins, "
          f"pitch {pitch:.1f} mm, housing Ø{p.housing_od:.0f}) ===")
    eta_needle = predict("needle", params_to_contacts(p)[1])
    print(f"  needle ring pins -> predicted η = {eta_needle*100:.0f}% regardless of bearing size "
          f"(rolling is rolling). Size is the real tradeoff:")
    print(f"  {'bearing':32s} {'rollerOD':>8s} {'min driveØ':>11s}  drop-in at Ø{p.pin_dia:.0f}?")
    for name, bore, od, w in NEEDLE_BEARINGS:
        d_pc, housing = min_drive_for_pin_od(p, od)
        dropin = "yes" if od <= p.pin_dia else f"no — grow to Ø{housing:.0f}"
        print(f"  {name:32s} {od:7.1f}  {housing:9.0f}    {dropin}")
    print(f"\n  => only the Ø2.5 loose-needle build replaces the current Ø{p.pin_dia:.0f} pins as-is."
          f"\n     Drawn-cup HK bearings (Ø8-9) force the drive from Ø{p.housing_od:.0f} up to ~Ø"
          f"{min_drive_for_pin_od(p, 8.0)[1]:.0f}-{min_drive_for_pin_od(p, 9.0)[1]:.0f} mm.")
    print()


def load_curve(p: Params):
    """η vs output torque, normalized to the no-load drag (motor-agnostic view)."""
    eta_inf = predict_params(p)
    d = drag_out(p)
    print(f"=== Load-dependent efficiency (η∞={eta_inf*100:.0f}%, "
          f"drag_out={d*1000:.0f} mN·m @ {p.ratio}:1) ===")
    print(f"  {'T_out (N·m)':>11s}  {'η':>5s}   (η rises toward η∞ as load grows)")
    for T in (0.02, 0.05, 0.10, 0.20, 0.40, 0.76):
        print(f"  {T:11.2f}  {eta_at_load(T, eta_inf, d)*100:4.0f}%")
    print()


if __name__ == "__main__":
    p = Params()
    ring, out = params_to_contacts(p)
    loss_map()
    config_table()
    bearing_sweep(p)
    load_curve(p)
    print(f"current config ({p.pin_mode} pins / {p.out_mode} output) -> "
          f"contacts [{ring} / {out}] -> η∞ = {predict_params(p)*100:.0f}% "
          f"(calibration-pending)")
