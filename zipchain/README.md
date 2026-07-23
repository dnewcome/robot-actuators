# Zip-chain linear actuator — research + Layer-B model

*Built 2026-07-22: `zipchain.py` (model) + `cad.py` + `sim.py`. This is **one of two
separate zipper-for-rigidity mechanisms** in the repo — the **LINEAR ACTUATOR**. Its
sibling, the **Y-zipper variable-stiffness element**, is a **different mechanism for a
different purpose** and lives in [`../yzipper/`](../yzipper/). They share only the
"mesh to make rigidity" trick; as devices they do different jobs — don't conflate them.*

| | **zip chain** (here) | **Y-zipper** ([`../yzipper/`](../yzipper/)) |
|---|---|---|
| strips | **2** | **3** |
| output | linear **thrust + stroke** (motion) | a **stiffness state** (flex↔rigid) |
| section | open, needs a guide | closed triangle, self-supporting |
| role | a linear **actuator** | a variable-stiffness / deployable **structure** |
| binding constraint | column **buckling** | tooth engagement / section stiffness |

---

## The mechanism (Tsubaki reference)

Two chain strands are fed into a sprocket head where they **interlock like a zipper**
into a single rigid pillar; run the sprocket backward and they separate and re-coil.
A normal chain only pulls — this one **pushes and pulls**, because the meshed column
carries compression, and the mesh **self-locks** so it holds a load with the motor off.

**Why it's genuinely different** from a ball screw / rack-and-pinion:
- **Compact storage → long stroke.** The strands store as two coils and deploy into a
  rigid strut (Tsubaki claims up to ~90% less floor area for a given stroke).
- **Push *and* pull** over the full stroke (rigid chain, not a cable).
- **Self-locking hold** at any position **without power** — ball screws can't.
- **Cheap to run** — < 1/30 the power of a comparable pneumatic/hydraulic cylinder.

Reference specs (industrial Zip Chain Actuator, medium/large): **38.2 kN**, **5,000 mm**
stroke, **1,000 mm/s**, **~90%** efficient with dedicated sprockets, 6M round trips,
multi-point stop, vertical-lift / horizontal-push / vertical-hang mounting.

## Why 2 strips — and why 3 is a different animal

Two strands zipped make an **open, near-flat section**: stiff to **in-plane** bending
only, weak out-of-plane and in torsion. Tsubaki gets away with it because the sprocket
**housing guides the weak axis** — it's a *guided push rod*. Add a third strip at 120°
and you get a **closed triangular tube** — rigid in every direction plus torsion, and
**self-supporting** with no guide. That extra rigidity is a different capability for a
different job (a stiffenable structure, not a push actuator) → it's the [`yzipper/`](../yzipper/)
thread, not this one.

## Fit to this repo

A **linear-actuator** cousin to `linear/` and `linear-rail-servo/`, but with a *different*
binding constraint — **column buckling**, not fluid compressibility or belt backlash. It
is **non-backdrivable with an energy-free hold**, which *fights* the repo's
quiet/compliant/backdrivable theme (like cycloidal does) yet **wins when the need is a
compact linear push that holds position with the motor off.** PLA-printable.

## The Layer-B model (BUILT — `zipchain.py`)

> **The binding constraint is COLUMN BUCKLING, not the motor.** The deployed free length
> is an Euler strut, so push capacity **falls as it extends**: `F_cr(L)=π²·E·I_col/(k·L)²`.
> Below a knee length **L\*** the actuator is drive/mesh-limited (flat); beyond L\* it is
> buckling-limited (∝1/L²). The push–stroke envelope is therefore **asymmetric** from the
> pull side (tension-limited by strand strength). `zipchain.py` exposes that envelope, a
> feasibility validator, and `out/envelope.png`.

Three design levers, all `Param`s: **column stiffness E·I** (section × engagement
fraction), **sprocket pitch radius r_p** (the gear ratio — trades thrust for deploy
speed, power-conserving), **end fixity k** (k=2 fixed-free cantilever conservative;
guided load end → k=1 quadruples F_cr).

**Default design results** (robot-scale, printed; calibration-pending strengths):

| Quantity | Value |
|---|---|
| deploy speed | **607 mm/s** @ 300 rpm (r_p = 19.3 mm, z=12) |
| drive thrust (τ/r_p) | **93 N** @ 2.0 N·m |
| push envelope | **93 N flat → 39 N** at full 320 mm extension |
| **buckling knee L\*** | **207 mm** (last ~⅓ of stroke is buckling-limited) |
| pull | 93 N flat (drive-limited; strand cap 1500 N) |
| hold @ full ext, motor off | **39 N** (mesh self-locks) |
| stored footprint | Ø81 mm for 300 mm stroke (store/stroke **0.27**) |

**`cad.py`** — 3 watertight solids: `strand` (a comb: outer spine + interlacing fingers),
`sprocket` (bored disc + z teeth), `head` (guide/merge housing that constrains the weak
axis). Two mirrored/half-pitch-staggered strands interlace into the deployed column.

**`sim.py`** — kinematic MuJoCo viewer: sprocket spin geared to column extension
(x=r_p·θ), column fed through the head; the montage annotates each frame with the push
capacity at that extension, so you watch the buckling ceiling fall as it deploys.
`make zipchain / zipchain-cad / sim-zipchain`.

---

## Sources
- Tsubaki Zip Chain (technology): https://en.tt-net.tsubakimoto.co.jp/tecs/pdct/sad/feat/pdct_feat_sad_ZC.asp
- Tsubaki Zip Chain Actuator (product): https://www.ustsubaki.com/products/zip-chain-actuator/
