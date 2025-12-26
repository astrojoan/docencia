#!/usr/bin/env python3
"""
2D perfectly elastic collision between two smooth disks (no friction).
Creates TWO animations:
  - LAB frame: positions r_i(t)
  - CM frame : positions r_i(t) - r_cm(t)  (so CM is fixed at origin)

Collision is handled analytically (single collision event):
We solve for t such that |(r2-r1) + (v2-v1)t| = (R1+R2), then update velocities
using the standard elastic impulse along the line of centers.

Deps:
  pip install numpy matplotlib pillow

Outputs:
  collision2d_lab.gif
  collision2d_cm.gif
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle


# -----------------------
# EASY-EDIT DEFAULTS
# -----------------------
CONFIG = dict(
    m1=1.0, m2=2.0,
    R1=0.25, R2=0.25,

    # Initial positions (meters, arbitrary units)
    x1=-2.5, y1=0.0,
    x2= 0.0, y2=0.0,

    # Initial velocities (m/s, arbitrary units)
    # Choose an "impact parameter" by setting y1 != y2 and vx1>0 to get a scattering angle.
    vx1= 2.5, vy1= 0.2,
    vx2= 0.0, vy2= 0.0,

    T=3.5,   # total duration (s)
    fps=30,  # frames per second

    lab_out="collision2d_lab.gif",
    cm_out ="collision2d_cm.gif",
)


def solve_collision_time(r1, r2, v1, v2, R):
    """
    Solve |(r2-r1) + (v2-v1)t|^2 = R^2 for t>=0.
    Returns smallest valid collision time, or None.
    """
    d0 = r2 - r1
    dv = v2 - v1

    a = np.dot(dv, dv)
    b = 2.0 * np.dot(d0, dv)
    c = np.dot(d0, d0) - R**2

    # If relative speed is ~0, no collision unless already overlapping
    if a < 1e-14:
        return 0.0 if c <= 0 else None

    disc = b*b - 4*a*c
    if disc < 0:
        return None

    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2*a)
    t2 = (-b + sqrt_disc) / (2*a)

    # candidate times
    candidates = [t for t in (t1, t2) if t >= 0]
    if not candidates:
        return None

    tc = min(candidates)

    # Check that they are actually approaching at collision (distance decreasing)
    d_at = d0 + dv * tc
    approaching = np.dot(d_at, dv) < 0  # derivative of |d|^2 is 2 d·dv
    if not approaching and c > 0:
        # if not approaching and they weren't overlapping initially, reject
        return None

    return tc


def elastic_update(v1, v2, r1, r2, m1, m2):
    """
    Update velocities for a perfectly elastic collision of smooth disks.
    Impulse along the unit normal n (line of centers).
    """
    d = r2 - r1
    dist = np.linalg.norm(d)
    if dist < 1e-14:
        return v1, v2  # degenerate; shouldn't happen

    n = d / dist  # from 1 to 2
    u = v1 - v2
    u_n = np.dot(u, n)

    # If separating, don't apply impulse
    if u_n <= 0:
        return v1, v2

    v1p = v1 - (2*m2/(m1+m2)) * u_n * n
    v2p = v2 + (2*m1/(m1+m2)) * u_n * n
    return v1p, v2p


def make_piecewise_trajectory(m1, m2, R1, R2, r1_0, r2_0, v1_0, v2_0, T, fps):
    """
    Generate r1(t), r2(t), v1(t), v2(t) for 0..T with a single analytic collision event if it occurs.
    """
    nframes = int(np.round(T * fps)) + 1
    t = np.linspace(0.0, T, nframes)

    R = R1 + R2
    tc = solve_collision_time(r1_0, r2_0, v1_0, v2_0, R)

    # Pre-allocate
    r1 = np.zeros((nframes, 2))
    r2 = np.zeros((nframes, 2))
    v1 = np.zeros((nframes, 2))
    v2 = np.zeros((nframes, 2))

    if tc is None or tc > T:
        # No collision in window
        r1[:] = r1_0 + np.outer(t, v1_0)
        r2[:] = r2_0 + np.outer(t, v2_0)
        v1[:] = v1_0
        v2[:] = v2_0
        return t, r1, r2, v1, v2, None

    # Positions at collision
    r1_c = r1_0 + v1_0 * tc
    r2_c = r2_0 + v2_0 * tc

    # Post-collision velocities
    v1_post, v2_post = elastic_update(v1_0, v2_0, r1_c, r2_c, m1, m2)

    pre = t <= tc
    post = ~pre

    # Pre
    r1[pre] = r1_0 + np.outer(t[pre], v1_0)
    r2[pre] = r2_0 + np.outer(t[pre], v2_0)
    v1[pre] = v1_0
    v2[pre] = v2_0

    # Post
    dt = (t[post] - tc)
    r1[post] = r1_c + np.outer(dt, v1_post)
    r2[post] = r2_c + np.outer(dt, v2_post)
    v1[post] = v1_post
    v2[post] = v2_post

    return t, r1, r2, v1, v2, tc


def animate_2d(t, r1, r2, v1, v2, m1, m2, R1, R2, out_gif, title, cm_frame=False):
    """
    Create a 2D animation GIF.
    If cm_frame=True, plot coordinates relative to r_cm(t), so CM stays at origin.
    """
    M = m1 + m2
    r_cm = (m1 * r1 + m2 * r2) / M
    V_cm = (m1 * v1 + m2 * v2) / M  # constant in ideal case

    if cm_frame:
        R1t = r1 - r_cm
        R2t = r2 - r_cm
        cm_points = np.zeros_like(r_cm)  # CM at (0,0)
    else:
        R1t = r1
        R2t = r2
        cm_points = r_cm

    # Determine axis limits with padding
    allx = np.concatenate([R1t[:, 0], R2t[:, 0], cm_points[:, 0]])
    ally = np.concatenate([R1t[:, 1], R2t[:, 1], cm_points[:, 1]])
    pad = 2.5 * max(R1, R2)

    xmin, xmax = allx.min() - pad, allx.max() + pad
    ymin, ymax = ally.min() - pad, ally.max() + pad

    # If CM frame, nice-ish symmetric limits
    if cm_frame:
        half = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
        xmin, xmax, ymin, ymax = -half, half, -half, half

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x (CM)" if cm_frame else "x (LAB)")
    ax.set_ylabel("y (CM)" if cm_frame else "y (LAB)")

    # Draw trajectories (faint)
    ax.plot(R1t[:, 0], R1t[:, 1], linewidth=1, alpha=0.25)
    ax.plot(R2t[:, 0], R2t[:, 1], linewidth=1, alpha=0.25)

    # Disks as Circle patches
    c1 = Circle((R1t[0, 0], R1t[0, 1]), R1, fill=False, linewidth=2)
    c2 = Circle((R2t[0, 0], R2t[0, 1]), R2, fill=False, linewidth=2)
    ax.add_patch(c1)
    ax.add_patch(c2)

    # CM marker
    cm_sc = ax.scatter([cm_points[0, 0]], [cm_points[0, 1]], marker="x", s=60)

    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    def update(i):
        c1.center = (R1t[i, 0], R1t[i, 1])
        c2.center = (R2t[i, 0], R2t[i, 1])
        cm_sc.set_offsets([[cm_points[i, 0], cm_points[i, 1]]])

        info.set_text(
            f"t = {t[i]:.2f} s\n"
            f"v1 = ({v1[i,0]:.2f}, {v1[i,1]:.2f})\n"
            f"v2 = ({v2[i,0]:.2f}, {v2[i,1]:.2f})\n"
            f"Vcm = ({V_cm[i,0]:.2f}, {V_cm[i,1]:.2f})"
        )
        return c1, c2, cm_sc, info

    dt = t[1] - t[0] if len(t) > 1 else 1.0 / 30.0
    anim = FuncAnimation(fig, update, frames=len(t), interval=1000 * dt, blit=False)

    anim.save(out_gif, writer=PillowWriter(fps=int(round(1.0 / dt))))
    plt.close(fig)


def parse_args():
    ap = argparse.ArgumentParser(description="2D elastic collision: LAB + CM GIFs.")
    # masses & radii
    ap.add_argument("--m1", type=float, default=CONFIG["m1"])
    ap.add_argument("--m2", type=float, default=CONFIG["m2"])
    ap.add_argument("--R1", type=float, default=CONFIG["R1"])
    ap.add_argument("--R2", type=float, default=CONFIG["R2"])
    # initial positions
    ap.add_argument("--x1", type=float, default=CONFIG["x1"])
    ap.add_argument("--y1", type=float, default=CONFIG["y1"])
    ap.add_argument("--x2", type=float, default=CONFIG["x2"])
    ap.add_argument("--y2", type=float, default=CONFIG["y2"])
    # initial velocities
    ap.add_argument("--vx1", type=float, default=CONFIG["vx1"])
    ap.add_argument("--vy1", type=float, default=CONFIG["vy1"])
    ap.add_argument("--vx2", type=float, default=CONFIG["vx2"])
    ap.add_argument("--vy2", type=float, default=CONFIG["vy2"])
    # timing
    ap.add_argument("--T", type=float, default=CONFIG["T"])
    ap.add_argument("--fps", type=int, default=CONFIG["fps"])
    # outputs
    ap.add_argument("--lab_out", type=str, default=CONFIG["lab_out"])
    ap.add_argument("--cm_out", type=str, default=CONFIG["cm_out"])
    return ap.parse_args()


def main():
    args = parse_args()

    m1, m2 = args.m1, args.m2
    R1, R2 = args.R1, args.R2

    r1_0 = np.array([args.x1, args.y1], dtype=float)
    r2_0 = np.array([args.x2, args.y2], dtype=float)
    v1_0 = np.array([args.vx1, args.vy1], dtype=float)
    v2_0 = np.array([args.vx2, args.vy2], dtype=float)

    t, r1, r2, v1, v2, tc = make_piecewise_trajectory(
        m1, m2, R1, R2, r1_0, r2_0, v1_0, v2_0, args.T, args.fps
    )

    if tc is None:
        print("Aviso: no hay colisión dentro del intervalo T (aun así genero los GIFs).")
    else:
        print(f"Colisión a t = {tc:.3f} s")

    animate_2d(
        t, r1, r2, v1, v2, m1, m2, R1, R2,
        out_gif=args.lab_out,
        title="Colisió elàstica 2D — LAB",
        cm_frame=False,
    )

    animate_2d(
        t, r1, r2, v1, v2, m1, m2, R1, R2,
        out_gif=args.cm_out,
        title="Colisió elàstica 2D — CM",
        cm_frame=True,
    )

    print(f"OK: {args.lab_out}")
    print(f"OK: {args.cm_out}")


if __name__ == "__main__":
    main()
