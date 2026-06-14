# Robot Actuators

Designing my own actuators for robot arms (and, longer term, broader robotics /
automation) — and, just as importantly, building a **simulate-first pipeline** so a
parametric CAD design flows straight into a physics sim before anything gets printed
or wired up.

The north star: change a parameter, re-run, and immediately see both the geometry
*and* the predicted torque / speed / inertia behavior — fast enough to screen many
designs without touching a dyno.

---

## The idea

I've used other people's actuator designs (cycloidal drives, etc.) and want to design
my own. The broad wish-list spans very different physics:

- Dyneema capstan / cable drives
- Hybrid electric-hydraulic linear actuators
- Linear brushless motors
- BLDC motors built for servo use (quasi-direct-drive)
- Cycloidal, wave, and harmonic (strain-wave) drives
- Compliant / gearless drives
- Exotic "muscle" actuators — SMA / memory wire, wax motors / heat drives

**Prior experience:** used others' cycloidal drives; experimented with linear actuators
including **peristaltic pumps driving cylinders**; experimented with **wax motors / heat
drives** and **SMA "muscles"** for silent actuation (SMA judged not viable alone, but a
useful jumping-off point).

### The organizing theme

The list isn't "actuators broadly" — the through-line is **quiet, compliant, low/no-gear
actuation for arms.** That filter sorts the list onto a spectrum:

```
  buildable / proven  ─────────────────────────────►  exotic / risky
  Dyneema capstan      QDD BLDC servo        SMA / wax muscle
  (silent, backdrivable,  (force-transparent,   (truly gearless & silent,
   ~95%/stage, light)      low/no gearing)       low η, slow, hard to control)
```

- **Fits the theme:** capstan, QDD-BLDC, exotic muscles.
- **Fights it:** cycloidal / harmonic / wave (geartrains — high torque density but the
  thing we're trying to avoid for silence/compliance) and electro-hydraulic (noisy/messy).
- Cycloidal still wins **when compactness is the binding constraint** — which is exactly
  the case for the current build.

---

## Simulate-first philosophy: two layers

The hard-won insight that shapes this whole repo:

> **MuJoCo cannot tell you the efficiency of a drive.** It's a rigid-body engine for
> *control and whole-system behavior*. When you give a joint a gear ratio and a friction
> loss, you are **typing in** the efficiency, not discovering it. The 70%-vs-90% question
> lives one layer below where MuJoCo operates.

So the problem splits cleanly:

**Layer A — "Does this actuator make the arm work?" → MuJoCo.**
Model each joint with a parametric actuator (gear ratio, reflected inertia, mass, a
load-dependent efficiency, and a back-EMF torque-speed curve). Output: the torque/speed/mass
envelope each joint needs. This is what MuJoCo is good at, and it's wired up today (`mujoco/`).

**Layer B — "Can this drive beat the efficiency ceiling?" → analytical, NOT a sim.**
Capstan efficiency = capstan equation + cable-bend hysteresis + bearing drag; gear
efficiency = mesh-friction formulas. These evaluate in microseconds — faster and more
trustworthy than any sim for *efficiency*. Escalate to FEA / multibody contact only near a
margin. A digital torque meter validates a single point. **Layer B v0 exists**
(`cycloidal/efficiency.py`): it predicts η per contact strategy and feeds that number into
the sim — so the MuJoCo η is no longer a hand-typed guess. The model's coefficients are
*calibration-pending* (one torque-meter point pins them down); the relative ranking of
strategies is already robust (see "Efficiency model" below).

CAD feeds both: mass/inertia → MuJoCo; key dimensions → the analytical model. And the loop
is closed: **Layer B predicts η → Layer A (the sim) consumes it**, all from the same Params.

---

## Repository layout

```
robot-actuators/
├── README.md
├── .venv/                     # python env (build123d, bd_warehouse, mujoco, numpy)
├── cycloidal/
│   ├── drive.py               # parametric build123d generator + feasibility validator (both stages)
│   ├── gears.py               # involute planetary gears: spur (sun/planet) + custom internal ring
│   ├── efficiency.py          # Layer B: two-stage η (planetary × cycloidal) + needle-bearing sweep
│   └── out/                   # generated .step + .stl for all 8 parts + assembly.step
└── mujoco/
    ├── actuator.py            # MotorSpec + ActuatorSpec: motor × total ratio → joint + servo numbers
    ├── testbench.xml          # Layer-A sizing scene (real meshes + load arm + payload)
    ├── run.py                 # injects physics + Layer-B η; runs sizing scenarios; --view demo
    └── viewer_scene.xml       # real-time kinematic demo of the full hybrid (both stages)
```

---

## Quickstart

```bash
# one-time setup
python3 -m venv .venv
.venv/bin/pip install build123d bd_warehouse mujoco numpy   # bd_warehouse = involute spur gears

# 1) generate CAD (edit Params in cycloidal/drive.py, then):
.venv/bin/python cycloidal/drive.py        # prints feasibility report, writes cycloidal/out/*

# 2) Layer-B efficiency model (η per contact strategy + needle-bearing fit sweep):
.venv/bin/python cycloidal/efficiency.py

# 3) actuator spec sheet (motor × gearbox -> joint numbers):
.venv/bin/python mujoco/actuator.py

# 4) Layer-A sizing sim (headless; uses the Layer-B η; prints lift / backdrive scenarios):
.venv/bin/python mujoco/run.py

# 5) look at the mechanism (real-time, needs a display):
.venv/bin/python mujoco/run.py --view
```

> Note: `source .venv/bin/activate` also works, but calling `.venv/bin/python` directly
> avoids the "Permission denied" you get from trying to *execute* `activate` instead of
> sourcing it.

---

## Current build: concentric hybrid planetary + cycloidal (40:1)

A fully **concentric compound**: the motor's high rpm goes through a **planetary first stage**,
whose carrier drives the **cycloidal eccentric**, whose disc drives the output — all on one
axis. Total reduction is the product of the two stages:

```
motor ─► PLANETARY 4:1 (sun in, ring fixed, carrier out) ─► cycloidal ECCENTRIC
      ─► CYCLOIDAL 10:1 (ring fixed, carrier out) ─► output flange      = 40:1
```

**Why split the ratio** (and put the cycloidal *last*):
- The planetary runs at high speed / low torque — the regime where gears are efficient — so a
  small ~97% stage buys 4× ratio cheaply and keeps the cycloidal **chunky** (fat rolling pins,
  healthy eccentricity) instead of forcing tiny high-lobe-count geometry to hit 40:1 alone.
- The cycloidal has the **lowest backlash**, so making it the output stage puts the precision
  where it counts; the planetary's lash divides by the cycloidal ratio when referred to output.

Single-stage cycloidal is still available — set `planetary=False` (ratio falls back to the lobe
count). Compactness is the binding constraint throughout, which is why this beats a capstan here.

### Motor — Goolsky / Surpass 2204 1400KV (measured)

| Spec | Value |
|---|---|
| Can / weight | Ø28 × 22 mm, ~28.7 g |
| Poles / config | 12N / 14P outrunner |
| Shaft | 3 mm, **exits on the mount side** |
| Mount | M3 cross pattern: one hole pair **16 mm** apart, the perpendicular pair **19 mm** apart |
| Electrical | 1400 KV, ~13 A burst, 0.307 Ω, 7–15 V (3S nominal) |

Because the shaft exits the mount side, the gearbox bolts to the motor base and the shaft
drives the eccentric directly — the arrangement the design assumed.

### Planetary stage parameters (all feasibility checks PASS)

| Param | Value | Notes |
|---|---|---|
| Ratio | **4.0:1** | `1 + n_ring/n_sun`; sun in, ring fixed, carrier out |
| Teeth (sun / planet / ring) | 12 / 12 / 36 | `n_ring = n_sun + 2·n_planet`; module 0.5 mm |
| Planets | 4 | equal-spaced: `(n_sun+n_ring) % n_planets == 0` |
| Gear Ø (PCD) | sun Ø6 / planet Ø6 / ring Ø18 | sun bores over the 3 mm motor shaft |
| Mesh quality | machined (~97%) | `planet_material` sets stage η (`printed` ~94%) |

Geometry is generated as real involute teeth (`cycloidal/gears.py`): external `SpurGear`s for
the sun and planets, and a custom involute **internal** ring gear (no library has one). The
validator checks tooth-count identity, equal-spacing, planet-to-planet collision, ring-fits-
housing, and sun-clears-shaft.

### Cycloidal stage parameters (all feasibility checks PASS)

| Param | Value | Notes |
|---|---|---|
| Ratio | **10:1** | lobe count; 11 ring pins, 10-lobe disc |
| Pin circle / housing OD | Ø32 / Ø39 mm | gearbox is larger than the can, as expected |
| Ring pins | 11 × Ø3 mm | steel dowels / cut rod |
| Eccentricity (E) | 0.70 mm | profile smoothness R/(E·N) = 2.08 (ideal) |
| Center bearing | 6700 (10×15×4) | gives a comfy 2.8 mm eccentric cam wall |
| Output pins | 6 × Ø2.5 mm | 0.85 mm clearance both sides of the lobe-root band |

### The feasibility validator — why it matters

Miniature cycloidals fail on **radial real estate**, not the lobe profile: the eccentric
cam must fit the 3 mm shaft + a wall + the offset + a bearing, and the output-pin circle
must live in the thin band between the center-bearing OD and the lobe root. `drive.py`
reports all those clearances **before** meshing geometry, so an infeasible design is caught
in milliseconds instead of in CAD or on the printer. (It already caught one bad config
during this build.)

Geometry is also checked, not just assumed: the generated disc has the correct **10 lobes**,
lobe depth **1.40 mm = 2E** exactly, and nests inside the Ø32 pin circle.

---

## 3D printing

Every run of `drive.py` writes **watertight STLs** (verified — slicers accept them) plus
STEP files for editing in real CAD:

| Part | STL bbox (mm) | Volume | Print notes |
|---|---|---|---|
| `ring.stl` | 39 × 39 × 7 | 4.6 cc | housing + motor mount; print flat, the pin pockets need clean walls |
| `disc.stl` | 30 × 30 × 5 | 2.0 cc | the cycloidal disc; print flat, fine layers (0.1 mm) for the lobe profile |
| `carrier.stl` | 26 × 26 × 6 | 1.1 cc | output flange; press-fit holes for steel pins (or integral pins in `"printed"` mode) |
| `eccentric.stl` | 10 × 10 × 4 | 0.3 cc | the cam; thinnest wall ~2.8 mm — the highest-stress part |

**Bill of non-printed materials** (printed by `drive.py`'s `hardware_bom()`, reflects the chosen contact modes):
- 11 × Ø3 mm steel dowel (ring pins — **free-spinning** rollers in the default `pin_mode="rolling"`)
- 6 × Ø2.5 mm steel dowel (output pins — pressed into the carrier in the default `out_mode="steel"`)
- 1 × **6700** bearing (10×15×4) for the eccentric
- **planetary gearset** — Ø6 sun + 4 × Ø6 planet + Ø18 internal ring, module 0.5 (machined/MOD
  gears for η, or SLA-printed for a prototype); 4 × planet idler pins on the carrier
- M3 screws for the motor cross-mount (16 mm + 19 mm pairs)
- the Goolsky 2204 motor

> The planetary gears (`sun.stl`, `planet.stl`, `planet_ring.stl`, `planet_carrier.stl`) are
> generated alongside the cycloidal parts. At module 0.5 they print on a 12k SLA but are the
> first candidates to *machine/buy* — small-module teeth are where print tolerance bites.

> **Status — printable geometry, not a finished gearbox.** This is a *kinematic / fit*
> prototype: the eight parts print and assemble to demonstrate the 40:1 hybrid mechanism, but
> it is **not yet a complete functional drive.** Still missing (see Roadmap): the output support
> bearing pocket, the motor pilot boss, the carrier↔eccentric coupling, gear tooth fillets /
> backlash tuning, and the output shaft / arm-joint interface. The small features (Ø2.5 holes,
> thin cam wall, module-0.5 teeth) will need a tolerance pass on your specific printer. PETG or
> a filled nylon is recommended for the disc and cam over PLA.

## Manufacturing strategy: print the structure, machine the contacts

A cycloidal's efficiency lives almost entirely in a few **contact interfaces**, not in the
bulk parts. So the rule this project follows:

- **3D print the structure** — ring/housing body, carrier body, motor adapter. These just
  locate things; a 12k SLA gives accurate bores and bolt patterns, and resin tackiness
  doesn't matter where nothing slides.
- **Don't print the contact surfaces.** The places that rub set η, and SLA ABS-like resin
  is tacky (high μ) and wears under Hertzian contact — the worst material exactly where it
  matters most. This is *why* "print the housing, machine/buy everything else" is the right
  instinct.

The contacts, in order of efficiency leverage:

| Interface | Carries | v1 default | Efficiency upgrade |
|---|---|---|---|
| Eccentric ↔ disc center | radial reduction load | **6700 bearing** (already rolling) | — |
| Ring pin ↔ disc lobe | main reduction torque | fixed Ø3 steel dowel (**sliding**) | **rotating roller**: dowel core + hardened sleeve / needle bearing → rolling |
| Output pin ↔ disc hole | output torque | integral printed pin (sliding) | steel pin + **bronze / IGUS bushing or roller** |

**Biggest single lever: make the ring pins rotate.** A fixed pin *slides* against the lobe
(high loss); a pin (or sleeve) that *spins* turns that into rolling contact. This is most of
what separates a ~70% printed cycloidal from a ~90% one — largely independent of disc
material.

**The disc** is the other variable: SLA-print it for a prototype (12k gives a smooth lobe
profile, but tacky resin + wear cap durability and η), or step up to **machined POM/acetal**
(self-lubricating, low μ, cheap to machine) or metal for the real article. If the pins roll,
disc material matters mostly for *wear*, not friction.

This is also the **Layer B bridge**: which contacts roll vs slide, and the materials' μ, are
exactly the inputs the analytical efficiency model needs. The manufacturing choice and the
efficiency prediction are the same decision.

**This is now a `Param`, not a note.** `drive.py` encodes the contact strategy directly:

| Param | Options | Effect |
|---|---|---|
| `pin_mode` | `"fixed"` \| `"rolling"` (default) | press-fit dowel (sliding) vs free-spinning dowel in an oversized pocket (rolling) |
| `pin_core_dia` | `≤ pin_dia` | set `< pin_dia` for a hardened sleeve on a thin core (validator checks the sleeve wall) |
| `out_mode` | `"printed"` \| `"steel"` (default) \| `"bushing"` | integral printed pins (sliding) → pressed steel pins (sliding, good surface) → core + rotating bushing (rolling) |

The validator prints the resulting **contact strategy** (`ROLLING`/`SLIDING` per interface)
and checks sleeve/bushing wall thicknesses; `hardware_bom()` lists exactly what to buy for
the chosen modes. True rolling bushings on the *output* pins are tight at Ø2.5 — the
pragmatic default is rolling ring pins + pressed steel output pins.

## Efficiency model (Layer B v0) — can rolling elements help?

`cycloidal/efficiency.py` is the analytical half of Layer B. It predicts η from the contact
strategy (friction-coefficient based, coefficients **calibration-pending** — one torque-meter
point tunes them) and feeds that number straight into the sim. Predicted η by strategy:

| Strategy | ring / output contact | predicted η |
|---|---|---|
| all printed (worst) | sliding / sliding | ~65% |
| steel pins, sliding | sliding / sliding | ~77% |
| **default: rolling pins + steel output** | rolling / sliding | **~86%** |
| sleeves both | rolling / rolling | ~90% |
| needle bearings everywhere | rolling / rolling | ~93% |

**The finding:** most of the gain (65 → 86%, **+21 points**) is already captured by the cheap
default — free-spinning ring pins. Going further to full needle bearings adds only **+4–7
points**, and at this size it's mechanically awkward:

```
needle-bearing fit sweep (current Ø3 pins, housing Ø39):
  Ø2.5 loose-needle build ...... drop-in (replaces current pins)
  Ø4 loose-needle .............. grow drive to Ø26
  HK0408 drawn cup (Ø8) ........ grow drive to Ø44
  HK0509 drawn cup (Ø9) ........ grow drive to Ø48
```

So: **rolling ring pins are the high-value move; needle bearings are diminishing returns**
that cost compactness — unless you build the marginal Ø2.5 loose-needle rollers, which fit
as-is. The model's *relative* ranking is robust (driven by μ ratios); the absolute numbers
wait on the torque meter.

## How it's wired (single source of truth)

```
cycloidal/drive.py  ──(ratio)──►  mujoco/actuator.py  ──►  mujoco/run.py
      │                                                         │
      └──► out/*.stl ─────────────────► testbench.xml ◄─────────┘
                                        viewer_scene.xml
```

The **total ratio** (`planet_ratio × cyclo_ratio`) is pulled straight out of `drive.py` into
`actuator.py`, and `run.py` injects the derived armature / gear / friction onto the MuJoCo model
at load time. **Change a `Param`, re-run, and the geometry and the sim move together** — they
can't drift apart. `p.ratio` is the single number the whole sim sees, so going hybrid changed
nothing downstream except the value.

`actuator.py` turns motor constants × ratio into joint-level numbers:
`Kt = 60/(2π·KV)`, `peak τ = η∞·Kt·I·N − drag`, `no-load ω = KV·V/N`, `reflected J = J_rotor·N²`,
and the back-EMF current limit `I = (throttle·V − Ke·ω)/R`. η∞ now chains both stages
(`η_planet × η_cyclo`) and `drag` refers through the full ratio — both from `efficiency.py`
(Layer B), so the whole chain is one source of truth.

### Positioning / servo resolution (BLDC "step counts")

A BLDC has no native steps like a stepper — but **the reduction multiplies whatever the rotor
resolves**, which is the lever for servo accuracy. For the 14-pole motor (7 pole-pairs) at 40:1:

| Rotor sensing | counts / motor rev | × 40 → output | output resolution | holds at standstill? |
|---|---|---|---|---|
| Hall / back-EMF 6-step | 6 × 7 = 42 | 1,680 | **0.214°** | Halls yes; back-EMF **no** |
| magnetic encoder 14-bit | 16,384 | 655,360 | ~0.001° | yes (FOC) |

Output **backlash ~0.08°** (planet lash ÷ cyclo ratio + tiny cyclo lash) sits below the Hall
resolution, so 0.214° is usable, not swamped by slop — that's *why* the cycloidal is last.
**Back-EMF amplitude ∝ speed**, so it gives commutation + velocity for efficient cruising but
**cannot hold position at rest**; pair it with Halls (cheap 0.214°) or an encoder (precise) for
the holding loop. `actuator.py`'s `servo_report()` prints this table for the current Params.

### Torque-based (load-dependent) efficiency

Efficiency isn't a flat number — a roughly constant **no-load drag** must be overcome before
useful torque appears, so η rises from 0 toward η∞ as load grows. This is realized in the sim
for free: **gear carries η∞ (0.83), joint frictionloss carries the drag (80 mN·m at output)**,
which reproduces `η(T) = η∞·T/(T+drag)`. The *operating* efficiency therefore tops out ~81% at
peak torque (peak load ≠ infinite load), not 83%. η∞ is itself the two-stage product
(planetary 0.97 × cyclo 0.86).

### Back-EMF and the torque-speed curve

The motor is now driven by **throttle (voltage)**, not raw current. At speed it generates
back-EMF (`Ke = Kt`), so the available current is `I = (throttle·V − Ke·ω) / R`, clamped to
the 13 A rating. That yields the real two-regime envelope:

- **current-limited** (flat ~2.87 N·m) below the corner speed (~249 rpm out), where the rating
  caps the current;
- **voltage-limited** above it, torque drooping linearly to **zero at the 388 rpm no-load
  speed** as back-EMF eats the headroom.

No more freewheeling: a free spin settles at no-load speed instead of accelerating forever.

---

## Simulation results (hybrid 40:1, load-dependent η, η∞ = 0.83)

```
total ratio ....... 40:1       (planetary 4.0 × cycloidal 10)
η∞ (high-load) .... 83%        (planetary 0.97 × cyclo 0.86); no-load drag 80 mN·m (output)
PEAK torque ....... 2.87 N·m   (@ 13 A burst, net of drag)
cont. torque ~..... 0.89 N·m   (thermal-limited estimate)
no-load out speed . 388 rpm
reflected inertia . 24e-4 kg·m²   (scales with ratio²)
max static payload  ~1.95 kg @ 150 mm (peak) / ~0.61 kg (continuous)
output resolution . 0.214° (Hall × 40)   backlash ~0.08°
```

**Scenario A — free accel:** output reaches ~30 rad/s in 50 ms; simulated effective
inertia (46.8e-4) **matches the hand calc** of arm+payload+armature → the model is
trustworthy.
**Scenario B — lift 100 g @ 150 mm:** needs 0.16 of 2.87 N·m available → lifts easily.
**Scenario C — unpowered hold:** backdrives under load → **backdrivable**, the desired
property for a compliant, safe arm.
**Scenario D — efficiency vs load (measured in sim):** η climbs 32% → 46% → 59% → 69% as
output torque rises 0.05 → 0.40 N·m, and the **measured η matches the model** — the load
curve emerges from the gear+frictionloss physics, not a typed-in constant.
**Scenario E — torque-speed curve (traced in a sim spin-up):** under full throttle the
delivered torque holds ~2.87 N·m to ~249 rpm, then droops toward zero at 388 rpm — the
back-EMF envelope, measured as the output accelerates (sample points scale with no-load speed).

The `--view` demo is a real-time kinematic scene of the **whole hybrid**: the cycloidal
**output** turns at 0.25 rev/s, the **planet carrier** at 10×, the **gold sun** at 40×, and the
**4 green planets** orbit (10×) while spinning on their pins (−20×) — the full 40:1 chain visible
at a glance.

---

## Roadmap

**The exciting frontier: a full electromechanical drive model.** Done so far: efficiency is
**load-dependent** (η rises with torque ✓), **geometry-driven** (predicted from the contact
strategy ✓ Layer B v0), and the motor now has **back-EMF + a real torque-speed curve** driven
by throttle/voltage with current limiting (✓). What's left:

- **Winding inductance (L) + fast current dynamics** — the steady-state electrical model (R,
  V, back-EMF, current limit) is in; add L and explicit FOC for transient current response.
- **Thermal** — I²R heating → the *real* continuous-torque limit instead of a 1/3 guess.
- **Load-dependent bearing losses** — extend Layer B so the drag itself grows with the mesh
  force (the eccentric bearing reacts the load), refining the low-load end of the η curve.

One CAD edit already predicts geometry → efficiency(load) → torque-speed behavior before any
hardware exists. Remaining items refine transients (L/FOC) and thermal limits.

**Nearer-term:**
- **Calibrate the Layer B model** with one torque-meter point (the v0 model exists and feeds
  the sim; the coefficients are still estimates). Extend it with load-dependent bearing losses.
- Drop this actuator into the real arm MJCF as a reusable joint module.
- Model the missing hardware: output support bearing pocket, motor pilot boss, output
  shaft / arm-joint interface (so it's actually printable).
- Add a second cycloidal disc at 180° (or a counterweight) to balance before high rpm.
- Sweep the total ratio: bump to **44:1** (cyclo 11 lobes) or **48:1** (cyclo 12), or shrink the
  Ø39 housing now that clearances are visible.
- Planetary detailing: tooth fillets, backlash/profile-shift for real print fit, and the
  carrier↔eccentric coupling (the two stages share that part).

**Done this iteration:** concentric **hybrid planetary + cycloidal (40:1)** — involute gears
generated (`gears.py`), two-stage η chained, total ratio flows through the whole sim, the full
compound animates in `--view`, and the BLDC **servo-resolution** model (ratio as a position
multiplier) is in `actuator.py`.

---

## Backlog / parked ideas

- QDD BLDC servo (Mini Cheetah / MIT style) — the modern quiet-arm path.
- SMA / wax "muscle" test rig — silent & gearless, high risk; research, not a deliverable.
- Electro-hydraulic linear actuators; peristaltic-pump-driven cylinders.
- Capstan / cable drive — the high-efficiency, silent play for when compactness isn't binding.
- Harmonic / wave drives; linear brushless motors.
- Broader "silent drive" survey.

---

## Caveats baked into the code

- **η is model-derived but calibration-pending.** Efficiency now comes from the Layer-B
  contact model (η∞ ≈ 0.86), is load-dependent, and feeds the sim — but the model's
  coefficients are estimates. One torque-meter point (no-load drag → `drag`; a loaded point →
  η∞) locks the absolute numbers. The *shape* (load-dependent, geometry-driven) is right; the
  *absolute* values aren't validated yet.
- **Electrical model is steady-state.** Back-EMF, resistance, bus voltage, and the current
  limit are in (real torque-speed curve), but winding inductance (L) and explicit FOC current
  dynamics are not — transient current response is idealized. Fine for sizing/torque-speed.
- **Continuous torque is a guess** (~1/3 of peak) until a thermal model replaces it.
- **The kinematic viewer treats the rotor as rigid** — the cycloidal disc's nutation isn't
  animated (it's a look-at-the-mechanism demo, not a contact simulation).
