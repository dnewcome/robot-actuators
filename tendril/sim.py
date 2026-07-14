"""
Kinematic MuJoCo viewer for the servo-driven continuum tendril.

The rigid parts (PLA mount, MG996R body, horn) load as the exported CAD meshes. The flexible TPU
finger genuinely DEFORMS, so it can't be a rigid STL — it's modelled as a tapered chain of hinge
segments driven to the exact curl shape κ(s) from tendril.py. The horn spins and two drive strings
run from the horn holes, through the deck guides, into the finger base — so you see: horn turns →
strings pull → finger curls. The viewer generates its own model (tendril/viewer.xml) each run so the
segment taper always matches TendrilParams.

    ../.venv/bin/python tendril/sim.py            # interactive curl sweep (± usable range)
    ../.venv/bin/python tendril/sim.py 0.4        # faster
    ../.venv/bin/python tendril/sim.py render      # headless -> out/sim.gif + montage
"""

import sys
import time
from math import pi
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tendril import TendrilParams
from cad import Cad

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
P = TendrilParams()
C = Cad(P)
N_SEG = 16                       # chain segments for the flexible finger


def _seg_dims():
    """Per-segment (thickness, width) half-sizes [m] at the arc midpoints."""
    seg_len = P.L / N_SEG
    dims = []
    for i in range(N_SEG):
        s_mid = (i + 0.5) * seg_len
        dims.append((P.t(s_mid) / 2, P.w(s_mid) / 2, seg_len / 2))
    return seg_len, dims


def build_xml():
    """Generate the viewer model (rigid meshes + tapered finger chain) and write viewer.xml."""
    seg_len, dims = _seg_dims()
    m = 1e-3
    z_base = (C.deck_top + C.flange_t) * m          # beam base (above the bolt flange)
    dbase = C.d_base * m
    rh = P.horn_r_mm * m
    boss = C.boss_h * m

    # nested finger chain, deepest last
    def chain(i):
        hx, hy, hz = dims[i]
        pos = f"0 0 {0.0 if i == 0 else seg_len:.5f}"
        sites = ""
        if i == 0:
            sites = (f'<site name="entry_p" pos="{dbase:.5f} 0 0" size="0.0007"/>'
                     f'<site name="entry_n" pos="{-dbase:.5f} 0 0" size="0.0007"/>')
        tip = f'<site name="tip" pos="0 0 {2*hz:.5f}" size="0.001"/>' if i == N_SEG - 1 else ""
        inner = chain(i + 1) if i < N_SEG - 1 else ""
        return (f'<body name="seg{i}" pos="{pos}">'
                f'<joint name="j{i}" type="hinge" axis="0 1 0"/>'
                f'<geom type="box" size="{hx:.5f} {hy:.5f} {hz:.5f}" pos="0 0 {hz:.5f}" '
                f'material="tpu"/>{sites}{tip}{inner}</body>')

    xml = f"""<mujoco model="tendril_viewer">
  <compiler meshdir="out" angle="radian" autolimits="true"/>
  <option gravity="0 0 0"/>
  <visual><global offwidth="960" offheight="720"/></visual>
  <asset>
    <mesh name="mount" file="mount.stl" scale="0.001 0.001 0.001"/>
    <mesh name="servo" file="servo.stl" scale="0.001 0.001 0.001"/>
    <mesh name="horn"  file="horn.stl"  scale="0.001 0.001 0.001"/>
    <material name="pla"    rgba="0.30 0.33 0.38 1"/>
    <material name="servo"  rgba="0.10 0.10 0.12 1"/>
    <material name="horn"   rgba="0.85 0.85 0.88 1"/>
    <material name="tpu"    rgba="0.90 0.45 0.15 1"/>
  </asset>

  <worldbody>
    <light pos="0.06 -0.06 0.16" dir="-0.3 0.3 -1"/>
    <camera name="iso" pos="0.10 -0.13 0.12" xyaxes="0.8 0.6 0 -0.30 0.40 0.87"/>

    <geom type="mesh" mesh="mount" material="pla"   contype="0" conaffinity="0"/>
    <geom type="mesh" mesh="servo" material="servo" contype="0" conaffinity="0"/>

    <!-- static bolt flange of the finger (rigid, bolted to the deck) -->
    <geom type="box" size="{C.flange_l/2*m:.5f} {C.flange_w/2*m:.5f} {C.flange_t/2*m:.5f}"
          pos="0 0 {(C.deck_top+C.flange_t/2)*m:.5f}" material="tpu" contype="0" conaffinity="0"/>

    <!-- deck string guides -->
    <site name="guide_p" pos="{dbase:.5f} 0 {C.deck_top*m:.5f}" size="0.0007"/>
    <site name="guide_n" pos="{-dbase:.5f} 0 {C.deck_top*m:.5f}" size="0.0007"/>

    <!-- servo horn (spins about Z) -->
    <body name="horn_body" pos="0 0 {boss:.5f}">
      <joint name="joint_horn" type="hinge" axis="0 0 1"/>
      <geom type="mesh" mesh="horn" material="horn" contype="0" conaffinity="0"/>
      <site name="hole_p" pos="{rh:.5f} 0 {C.horn_t*m:.5f}" size="0.0008"/>
      <site name="hole_n" pos="{-rh:.5f} 0 {C.horn_t*m:.5f}" size="0.0008"/>
    </body>

    <!-- flexible finger chain -->
    <body name="finger" pos="0 0 {z_base:.5f}">
      {chain(0)}
    </body>
  </worldbody>

  <tendon>
    <spatial name="t_p" width="0.0006" rgba="0.95 0.95 0.95 1">
      <site site="hole_p"/><site site="guide_p"/><site site="entry_p"/></spatial>
    <spatial name="t_n" width="0.0006" rgba="0.55 0.85 0.95 1">
      <site site="hole_n"/><site site="guide_n"/><site site="entry_n"/></spatial>
  </tendon>
</mujoco>"""
    (HERE / "viewer.xml").write_text(xml)
    return HERE / "viewer.xml"


def _seg_angles(psi):
    """Per-segment incremental bend angle Δθ_i from the tendril.py curl shape at horn angle ψ."""
    s, theta, _, _, _, _ = P.shape(psi)
    sb = np.linspace(0.0, P.L, N_SEG + 1)
    th_b = np.interp(sb, s, theta)
    return np.diff(th_b)


def _set(m, d, name, val):
    d.qpos[m.jnt_qposadr[m.joint(name).id]] = val


def _apply(m, d, psi):
    _set(m, d, "joint_horn", psi)
    for i, dth in enumerate(_seg_angles(psi)):
        _set(m, d, f"j{i}", dth)
    mujoco.mj_forward(m, d)


def view(cycles_per_s=0.2):
    from mujoco import viewer as mjv
    path = build_xml()
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    up = P.usable_psi
    dt = 1.0 / 60.0
    print(f"\nviewer: servo-driven tendril | sweeps ±{np.rad2deg(up):.0f}° horn "
          f"(±{P.usable_bend_deg:.0f}° curl, tension-limited)")
    print("  close window to exit")
    with mjv.launch_passive(m, d) as v:
        ph = 0.0
        while v.is_running():
            ph += 2 * pi * cycles_per_s * dt
            _apply(m, d, up * np.sin(ph))
            v.sync()
            time.sleep(dt)


def render(n_frames=48, height=720, width=960):
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    path = build_xml()
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    try:
        r = mujoco.Renderer(m, height=height, width=width)
    except Exception:
        os.environ["MUJOCO_GL"] = "osmesa"
        r = mujoco.Renderer(m, height=height, width=width)

    up = P.usable_psi
    frames = []
    for i in range(n_frames):
        _apply(m, d, up * np.sin(2 * pi * i / (n_frames - 1)))
        r.update_scene(d, camera="iso")
        frames.append(r.render().copy())

    OUT.mkdir(exist_ok=True)
    gif = OUT / "sim.gif"
    try:
        import imageio.v2 as imageio
        imageio.mimsave(gif, frames, fps=18, loop=0)
        print(f"  wrote {gif}")
    except Exception:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=55, loop=0)
        print(f"  wrote {gif}  (via PIL)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picks = [n_frames // 4, 3 * n_frames // 8, n_frames // 2, 5 * n_frames // 8,
             3 * n_frames // 4, 7 * n_frames // 8]
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    for a, k in zip(ax.ravel(), picks):
        psi = up * np.sin(2 * pi * k / (n_frames - 1))
        a.imshow(frames[k]); a.set_axis_off()
        a.set_title(f"horn {np.rad2deg(psi):+.0f}° → curl {P.bend_deg(psi):.0f}°", fontsize=9)
    fig.suptitle("Servo-driven continuum tendril — one MG996R curls the TPU finger both ways",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "sim_montage.png", dpi=110)
    print(f"  wrote {OUT/'sim_montage.png'}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "render":
        render()
    else:
        view(cycles_per_s=float(arg) if arg else 0.2)
