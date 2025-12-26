#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.widgets import Slider


# =========================
# CONSTANTS (edita ací)
# =========================
R = 0.25     # radi dels dos discos
L = 4.0      # separació inicial en x entre centres
T = 3.5      # durada total (s)
FPS = 40     # frames per segon


# ---------- Physics helpers ----------

def solve_collision_time(r1, r2, v1, v2, Rsum):
    d0 = r2 - r1
    dv = v2 - v1
    a = np.dot(dv, dv)
    b = 2.0 * np.dot(d0, dv)
    c = np.dot(d0, d0) - Rsum**2

    if a < 1e-14:
        return 0.0 if c <= 0 else None

    disc = b*b - 4*a*c
    if disc < 0:
        return None

    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2*a)
    t2 = (-b + sqrt_disc) / (2*a)

    candidates = [t for t in (t1, t2) if t >= 0]
    if not candidates:
        return None

    tc = min(candidates)

    # Must be approaching at collision unless already overlapping
    d_at = d0 + dv * tc
    approaching = np.dot(d_at, dv) < 0
    if not approaching and c > 0:
        return None

    return tc


def elastic_update(v1, v2, r1, r2, m1, m2):
    d = r2 - r1
    dist = np.linalg.norm(d)
    if dist < 1e-14:
        return v1, v2

    n = d / dist
    u = v1 - v2
    u_n = np.dot(u, n)

    # separating => no impulse
    if u_n <= 0:
        return v1, v2

    v1p = v1 - (2*m2/(m1+m2)) * u_n * n
    v2p = v2 + (2*m1/(m1+m2)) * u_n * n
    return v1p, v2p


def make_piecewise_trajectory(m1, m2, R, r1_0, r2_0, v1_0, v2_0, T, fps):
    nframes = int(np.round(T * fps)) + 1
    t = np.linspace(0.0, T, nframes)

    tc = solve_collision_time(r1_0, r2_0, v1_0, v2_0, 2*R)

    r1 = np.zeros((nframes, 2))
    r2 = np.zeros((nframes, 2))
    vv1 = np.zeros((nframes, 2))
    vv2 = np.zeros((nframes, 2))

    if tc is None or tc > T:
        r1[:] = r1_0 + np.outer(t, v1_0)
        r2[:] = r2_0 + np.outer(t, v2_0)
        vv1[:] = v1_0
        vv2[:] = v2_0
        return t, r1, r2, vv1, vv2, None

    r1_c = r1_0 + v1_0 * tc
    r2_c = r2_0 + v2_0 * tc
    v1_post, v2_post = elastic_update(v1_0, v2_0, r1_c, r2_c, m1, m2)

    pre = t <= tc
    post = ~pre

    r1[pre] = r1_0 + np.outer(t[pre], v1_0)
    r2[pre] = r2_0 + np.outer(t[pre], v2_0)
    vv1[pre] = v1_0
    vv2[pre] = v2_0

    dt = t[post] - tc
    r1[post] = r1_c + np.outer(dt, v1_post)
    r2[post] = r2_c + np.outer(dt, v2_post)
    vv1[post] = v1_post
    vv2[post] = v2_post

    return t, r1, r2, vv1, vv2, tc


def build_initial_conditions(L, b, v1mag, theta_deg, v2x):
    r1_0 = np.array([-L/2, b], dtype=float)
    r2_0 = np.array([ L/2, 0.0], dtype=float)

    th = np.deg2rad(theta_deg)
    v1_0 = np.array([v1mag*np.cos(th), v1mag*np.sin(th)], dtype=float)
    v2_0 = np.array([v2x, 0.0], dtype=float)
    return r1_0, r2_0, v1_0, v2_0


# ---------- Main (interactive) ----------

def main():
    # Reserve a bottom band for sliders (so plots never overlap them)
    fig, (ax_lab, ax_cm) = plt.subplots(1, 2, figsize=(11.5, 6.5))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.26, wspace=0.18)

    for ax in (ax_lab, ax_cm):
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    ax_lab.set_title("LAB")
    ax_cm.set_title("CM (centre de masses fix)")

    # Slider axes in the reserved bottom band
    # (y positions are in figure coordinates)
    def slider_axis(y):
        return fig.add_axes([0.10, y, 0.80, 0.03])

    # Only the sliders you want:
    s_b     = Slider(slider_axis(0.19), "b",     -2.0,  2.0,  valinit=0.6)
    s_theta = Slider(slider_axis(0.15), "theta", -80.0, 80.0, valinit=0.0, valfmt="%1.1f°")
    s_v1    = Slider(slider_axis(0.11), "v1",     0.0,  6.0,  valinit=2.5)
    s_v2x   = Slider(slider_axis(0.07), "v2x",   -3.0,  3.0,  valinit=0.0)
    s_m1    = Slider(slider_axis(0.03), "m1",     0.2,  5.0,  valinit=1.0)
    s_m2    = Slider(slider_axis(-0.01), "m2",    0.2,  5.0,  valinit=2.0)  # slightly lower but still visible

    # If m2 ends up too low on your screen, change -0.01 -> 0.00 and shift others up a bit.

    # Artists
    traj1_lab, = ax_lab.plot([], [], linewidth=1, alpha=0.25)
    traj2_lab, = ax_lab.plot([], [], linewidth=1, alpha=0.25)
    traj1_cm,  = ax_cm.plot([], [], linewidth=1, alpha=0.25)
    traj2_cm,  = ax_cm.plot([], [], linewidth=1, alpha=0.25)

    c1_lab = Circle((0, 0), R, fill=False, linewidth=2)
    c2_lab = Circle((0, 0), R, fill=False, linewidth=2)
    ax_lab.add_patch(c1_lab)
    ax_lab.add_patch(c2_lab)
    cm_lab = ax_lab.scatter([0], [0], marker="x", s=60)

    c1_cm = Circle((0, 0), R, fill=False, linewidth=2)
    c2_cm = Circle((0, 0), R, fill=False, linewidth=2)
    ax_cm.add_patch(c1_cm)
    ax_cm.add_patch(c2_cm)
    cm_cm = ax_cm.scatter([0], [0], marker="x", s=60)

    info = fig.text(0.5, 0.94, "", ha="center", va="top")

    state = dict(t=None, r1=None, r2=None, v1=None, v2=None, tc=None, rcm=None, r1_cm=None, r2_cm=None)
    idx = {"i": 0}

    def set_limits(ax, X1, X2, R, symmetric=False):
        allx = np.concatenate([X1[:, 0], X2[:, 0]])
        ally = np.concatenate([X1[:, 1], X2[:, 1]])
        pad = 2.8 * R
        xmin, xmax = allx.min() - pad, allx.max() + pad
        ymin, ymax = ally.min() - pad, ally.max() + pad
        if symmetric:
            half = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
            xmin, xmax, ymin, ymax = -half, half, -half, half
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    def recompute(_=None):
        b = s_b.val
        theta_deg = s_theta.val
        v1mag = s_v1.val
        v2x = s_v2x.val
        m1 = s_m1.val
        m2 = s_m2.val

        r1_0, r2_0, v1_0, v2_0 = build_initial_conditions(L, b, v1mag, theta_deg, v2x)
        t, r1, r2, vv1, vv2, tc = make_piecewise_trajectory(m1, m2, R, r1_0, r2_0, v1_0, v2_0, T, FPS)

        M = m1 + m2
        rcm = (m1 * r1 + m2 * r2) / M
        r1_cm = r1 - rcm
        r2_cm = r2 - rcm

        state.update(t=t, r1=r1, r2=r2, v1=vv1, v2=vv2, tc=tc, rcm=rcm, r1_cm=r1_cm, r2_cm=r2_cm)

        # Update trajectories
        traj1_lab.set_data(r1[:, 0], r1[:, 1])
        traj2_lab.set_data(r2[:, 0], r2[:, 1])
        traj1_cm.set_data(r1_cm[:, 0], r1_cm[:, 1])
        traj2_cm.set_data(r2_cm[:, 0], r2_cm[:, 1])

        # Update limits
        set_limits(ax_lab, r1, r2, R, symmetric=False)
        set_limits(ax_cm, r1_cm, r2_cm, R, symmetric=True)

        # Info line
        if tc is None:
            info.set_text("⚠️ No hi ha col·lisió dins de T (prova: puja v1, baixa |b|, o canvia theta).")
        else:
            info.set_text(f"Col·lisió a t = {tc:.3f} s")

        idx["i"] = 0
        fig.canvas.draw_idle()

    def update(_):
        t = state["t"]
        if t is None or len(t) == 0:
            return []

        i = idx["i"]
        if i >= len(t):
            idx["i"] = 0
            i = 0

        r1 = state["r1"][i]
        r2 = state["r2"][i]
        rcm = state["rcm"][i]
        r1c = state["r1_cm"][i]
        r2c = state["r2_cm"][i]

        c1_lab.center = (r1[0], r1[1])
        c2_lab.center = (r2[0], r2[1])
        cm_lab.set_offsets([[rcm[0], rcm[1]]])

        c1_cm.center = (r1c[0], r1c[1])
        c2_cm.center = (r2c[0], r2c[1])
        cm_cm.set_offsets([[0.0, 0.0]])

        idx["i"] += 1
        return [c1_lab, c2_lab, c1_cm, c2_cm]

    # Build animation
    anim = FuncAnimation(fig, update, interval=1000 / FPS, blit=False)

    # Initial compute + hook sliders
    recompute()
    for s in (s_b, s_theta, s_v1, s_v2x, s_m1, s_m2):
        s.on_changed(recompute)

    plt.show()


if __name__ == "__main__":
    main()
