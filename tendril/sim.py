"""
Kinematic MuJoCo viewer for the segmented servo-driven tendril.

Rigid parts (PLA mount, MG996R body, horn) load as the exported CAD meshes. The flexible finger is
a chain of RIGID VERTEBRA boxes joined by hinges at the spine gaps — the real mechanism — driven to
the per-joint bend from tendril.py. The servo shaft runs ACROSS the finger axis, so the horn spins
in the bending plane and the two drive strings drop straight down onto it (no 90° cable bend). The
viewer generates its own model (tendril/viewer.xml) each run so the segment layout matches params.

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


def build_xml():
    """Generate the viewer model (rigid meshes + vertebra hinge-chain) and write viewer.xml."""
    m = 1e-3
    seg = P.seg_len_mm * m
    gap = P.gap_mm * m
    pitch = seg + gap
    hx, hy, hz = P.seg_t_mm / 2 * m, P.seg_w_mm / 2 * m, seg / 2
    d = P.d_off_mm * m
    rh = P.horn_r_mm * m
    yh = C.y_horn * m
    z_horn = C.z_horn * m
    z0 = (C.shelf_top + C.flange_t) * m           # base of vertebra 0 (above the bolt flange)

    # finger: vertebra 0 welded to the mount, then a hinge (about Y) at each spine gap
    def chain(i):
        joint = "" if i == 0 else f'<joint name="j{i-1}" type="hinge" axis="0 1 0" pos="0 0 {-gap/2:.5f}"/>'
        pos = f"0 {yh:.5f} {z0 + hz:.5f}" if i == 0 else f"0 0 {pitch:.5f}"
        sites = (f'<site name="entry_p" pos="{d:.5f} 0 {-hz:.5f}" size="0.0008"/>'
                 f'<site name="entry_n" pos="{-d:.5f} 0 {-hz:.5f}" size="0.0008"/>') if i == 0 else ""
        inner = chain(i + 1) if i < P.n_vert - 1 else ""
        return (f'<body name="v{i}" pos="{pos}">{joint}'
                f'<geom type="box" size="{hx:.5f} {hy:.5f} {hz:.5f}" material="tpu"/>'
                f'{sites}{inner}</body>')

    xml = f"""<mujoco model="tendril_seg_viewer">
  <compiler meshdir="out" angle="radian" autolimits="true"/>
  <option gravity="0 0 0"/>
  <visual><global offwidth="960" offheight="720"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <mesh name="mount" file="mount.stl" scale="0.001 0.001 0.001"/>
    <mesh name="servo" file="servo.stl" scale="0.001 0.001 0.001"/>
    <mesh name="horn"  file="horn.stl"  scale="0.001 0.001 0.001"/>
    <material name="pla"   rgba="0.32 0.35 0.40 1"/>
    <material name="servo" rgba="0.10 0.10 0.12 1"/>
    <material name="horn"  rgba="0.85 0.85 0.88 1"/>
    <material name="tpu"   rgba="0.90 0.45 0.15 1"/>
  </asset>

  <worldbody>
    <light pos="0.08 0.18 0.22" dir="-0.2 -0.4 -1"/>
    <camera name="iso" pos="0.16 0.16 0.12" xyaxes="-0.728 0.686 0 -0.130 -0.138 0.981"/>

    <geom type="mesh" mesh="mount" material="pla"   contype="0" conaffinity="0"/>
    <geom type="mesh" mesh="servo" material="servo" contype="0" conaffinity="0"/>

    <!-- static bolt flange of the finger (rigid, bolted to the shelf) -->
    <geom type="box" size="{C.flange_l/2*m:.5f} {C.flange_w/2*m:.5f} {C.flange_t/2*m:.5f}"
          pos="0 {yh:.5f} {(C.shelf_top+C.flange_t/2)*m:.5f}" material="tpu" contype="0" conaffinity="0"/>

    <!-- deck string guides (align to the channels) -->
    <site name="guide_p" pos="{d:.5f} {yh:.5f} {C.shelf_top*m:.5f}" size="0.0008"/>
    <site name="guide_n" pos="{-d:.5f} {yh:.5f} {C.shelf_top*m:.5f}" size="0.0008"/>

    <!-- servo horn: spins about Y in the bending plane -->
    <body name="horn_body" pos="0 {yh:.5f} {z_horn:.5f}">
      <joint name="joint_horn" type="hinge" axis="0 1 0"/>
      <geom type="mesh" mesh="horn" material="horn" contype="0" conaffinity="0"/>
      <site name="hole_p" pos="{rh:.5f} 0 0" size="0.0009"/>
      <site name="hole_n" pos="{-rh:.5f} 0 0" size="0.0009"/>
    </body>

    <!-- flexible finger: vertebra hinge-chain -->
    {chain(0)}
  </worldbody>

  <tendon>
    <spatial name="t_p" width="0.0007" rgba="0.95 0.95 0.95 1">
      <site site="hole_p"/><site site="guide_p"/><site site="entry_p"/></spatial>
    <spatial name="t_n" width="0.0007" rgba="0.55 0.85 0.95 1">
      <site site="hole_n"/><site site="guide_n"/><site site="entry_n"/></spatial>
  </tendon>
</mujoco>"""
    (HERE / "viewer.xml").write_text(xml)
    return HERE / "viewer.xml"


def _set(m, d, name, val):
    d.qpos[m.jnt_qposadr[m.joint(name).id]] = val


def _apply(m, d, psi):
    _set(m, d, "joint_horn", psi)
    dth = P.joint_angle(psi)
    for i in range(P.n_joints):
        _set(m, d, f"j{i}", dth)
    mujoco.mj_forward(m, d)


def view(cycles_per_s=0.2):
    from mujoco import viewer as mjv
    path = build_xml()
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    up = P.usable_psi
    dt = 1.0 / 60.0
    print(f"\nviewer: segmented tendril | sweeps ±{np.rad2deg(up):.0f}° horn "
          f"(±{P.usable_bend_deg:.0f}° curl)")
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
    fig.suptitle("Segmented servo-driven tendril — horizontal-shaft servo curls the vertebrae-on-spine "
                 "finger both ways", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "sim_montage.png", dpi=110)
    print(f"  wrote {OUT/'sim_montage.png'}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "render":
        render()
    else:
        view(cycles_per_s=float(arg) if arg else 0.2)
