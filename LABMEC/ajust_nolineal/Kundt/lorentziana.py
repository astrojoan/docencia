#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pràctica 6 — Tub de Kundt: ajust lorentzià del perfil de ressonància A(ω) + plots + export

Aquest script llig dades (freqüència vs amplitud) i ajusta una corba lorentziana.

Model (amplitud):
    A(ω) = B / sqrt( (ω_r^2 - ω^2)^2 + (2βω)^2 )

on:
- B     és proporcional a l'amplitud en ressonància
- ω_r   és la freqüència pròpia (ressonància)
- β     és el factor d'esmorteïment

IMPORTANT (unitats):
- Normalment mesures la freqüència del generador en Hz (ν).
- L'equació està en ω (rad/s): ω = 2π ν.
- Per defecte, aquest script assumeix que x està en Hz i fa la conversió.
  Si el teu CSV ja té ω, usa l'opció: --x-is-omega

Ús (mínim):
  python lorentziana.py dadesKundt.csv

Format CSV recomanat (com l'exemple):
  x, y, sigma_x, sigma_y
  x       : freqüència (Hz)   (o ω si uses --x-is-omega)
  y       : amplitud (p. ex. Vpp)
  sigma_x : error en x
  sigma_y : error en y

Si no tens columnes d'errors, també funciona amb només:
  x, y
(i aleshores fa un ajust no ponderat)

Eixida (automàtica):
- Crea una carpeta d'eixida (per defecte: resultats_lorentz/)
- Guarda figures PNG: <prefix>_fit.png, <prefix>_residus.png
- Exporta fitxers: <prefix>_resultats_fit.txt/csv, <prefix>_residus.csv,
                   <prefix>_corba_fit.csv, <prefix>_covariancia.csv, <prefix>_resultats.zip

Com executar-ho si NO tens Python instal·lat (forma fàcil, Windows/macOS/Linux):
1) Instal·la Miniconda (busca “Miniconda download”) amb les opcions per defecte.
2) Obri una terminal:
   - Windows: “Miniconda Prompt” o “Anaconda Prompt”
   - macOS: Terminal
   - Linux: Terminal
3) Crea un entorn i instal·la el que cal (una sola vegada):
     conda create -n labmec python=3.11 numpy scipy pandas matplotlib -y
     conda activate labmec
4) Ves a la carpeta on tens els fitxers i executa:
     python lorentziana.py dadesKundt.csv
"""

import argparse
import csv
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ------------------------------------------------------
# MODEL LORENTZIÀ (amplitud)
# ------------------------------------------------------
def lorentz_amp(omega, B, omega_r, beta):
    return B / np.sqrt((omega_r**2 - omega**2)**2 + (2.0 * beta * omega)**2)


PARAM_NAMES = ["B", "omega_r", "beta"]


# ------------------------------------------------------
# Seeds: estimació raonable a partir del pic i del nivell Amax/sqrt(2)
# ------------------------------------------------------
def estimate_seeds(omega, y):
    i_peak = int(np.argmax(y))
    omega_r0 = float(omega[i_peak])
    y_peak = float(y[i_peak])

    # Nivell Amax/sqrt(2) (com a la definició de l'amplària de ressonància)
    y_half = y_peak / np.sqrt(2.0)

    def interp_cross(j1, j2):
        x1, y1 = float(omega[j1]), float(y[j1])
        x2, y2 = float(omega[j2]), float(y[j2])
        if y2 == y1:
            return None
        t = (y_half - y1) / (y2 - y1)
        if 0.0 <= t <= 1.0:
            return x1 + t * (x2 - x1)
        return None

    omega_left = None
    for j in range(i_peak, 0, -1):
        if (y[j] - y_half) * (y[j - 1] - y_half) <= 0:
            omega_left = interp_cross(j, j - 1)
            if omega_left is not None:
                break

    omega_right = None
    for j in range(i_peak, len(y) - 1):
        if (y[j] - y_half) * (y[j + 1] - y_half) <= 0:
            omega_right = interp_cross(j, j + 1)
            if omega_right is not None:
                break

    if omega_left is not None and omega_right is not None and omega_right > omega_left:
        width = float(omega_right - omega_left)
        beta0 = max(width / 2.0, 1e-12)  # aproximació: (right-left) ~ 2β
    else:
        beta0 = max((float(omega.max() - omega.min())) / 30.0, 1e-12)

    # A(omega_r) = B / (2β ω_r)  =>  B ≈ A_peak * 2β ω_r
    B0 = float(y_peak * 2.0 * beta0 * omega_r0)
    return [B0, omega_r0, beta0]


# ------------------------------------------------------
# Derivada numèrica per incorporar error en X (sigma_eff)
# ------------------------------------------------------
def dAdw_numeric(omega, params, eps=1e-2):
    return (lorentz_amp(omega + eps, *params) - lorentz_amp(omega - eps, *params)) / (2.0 * eps)


def fit_with_sigma_eff(omega, y, sigma_y, sigma_omega, p0, bounds, n_iter=5):
    """
    Ponderem amb:
      sigma_eff^2 = sigma_y^2 + (dA/dω * sigma_omega)^2
    i iterem unes quantes vegades.
    """
    sigma = sigma_y.copy().astype(float)
    popt = None
    pcov = None

    for _ in range(int(n_iter)):
        popt, pcov = curve_fit(
            lorentz_amp,
            omega,
            y,
            p0=(p0 if popt is None else popt),
            sigma=sigma,
            absolute_sigma=True,
            bounds=bounds,
            maxfev=200000,
        )
        dA = dAdw_numeric(omega, popt, eps=1e-2)
        sigma = np.sqrt(sigma_y**2 + (dA * sigma_omega) ** 2)

    return popt, pcov, sigma


# ------------------------------------------------------
# Banda d'incertesa (MC covariància) per pintar-la al plot principal
# ------------------------------------------------------
def make_psd(cov, jitter=1e-15):
    cov = 0.5 * (cov + cov.T)
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 0.0, None)
    cov_psd = (V * w) @ V.T
    scale = float(np.max(np.diag(cov_psd))) if cov_psd.size else 1.0
    cov_psd = cov_psd + (jitter * (scale if scale > 0 else 1.0)) * np.eye(cov_psd.shape[0])
    return cov_psd


def mc_band_constrained(omega_grid, popt, pcov, bounds, n_draws=1200, seed=12345):
    rng = np.random.default_rng(seed)
    cov_psd = make_psd(np.array(pcov, dtype=float))
    lo_b = np.array(bounds[0], dtype=float)
    hi_b = np.array(bounds[1], dtype=float)

    accepted = []
    target = int(n_draws)
    batch = max(500, target)
    attempts = 0

    while len(accepted) < target and attempts < 50:
        draws = rng.multivariate_normal(mean=np.array(popt, float), cov=cov_psd, size=batch)
        attempts += 1

        ok = np.all(draws >= lo_b, axis=1) & np.all(draws <= hi_b, axis=1)
        draws = draws[ok]
        if draws.size == 0:
            continue

        for p in draws:
            yv = lorentz_amp(omega_grid, *p)
            if np.all(np.isfinite(yv)):
                accepted.append(yv)
            if len(accepted) >= target:
                break

    if len(accepted) == 0:
        return None, None

    ys = np.array(accepted[:target])
    lo = np.percentile(ys, 16, axis=0)
    hi = np.percentile(ys, 84, axis=0)
    return lo, hi


# ------------------------------------------------------
# Export
# ------------------------------------------------------
def export_all(outdir, prefix, x_label, x_vals, y_vals, sx_vals, sy_vals, sigma_eff,
               popt, perr, pcov, chi2, chi2_red, dof,
               x_grid, y_grid, lo, hi):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = []

    # TXT resum
    txt_path = outdir / f"{prefix}_resultats_fit.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Resultats del fit (Pràctica 6 — Tub de Kundt: perfil de ressonància)\n\n")
        f.write(f"chi2: {chi2}\n")
        f.write(f"chi2_red: {chi2_red}\n")
        f.write(f"dof: {dof}\n\n")
        f.write("Paràmetres (valor ± 1σ)\n")
        for name, val, err in zip(PARAM_NAMES, popt, perr):
            f.write(f"{name}: {val} +/- {err}\n")
    files.append(txt_path)

    # CSV paràmetres
    par_csv = outdir / f"{prefix}_resultats_fit.csv"
    with open(par_csv, "w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        wri.writerow(["parametre", "valor", "error_1sigma"])
        for name, val, err in zip(PARAM_NAMES, popt, perr):
            wri.writerow([name, float(val), float(err)])
        wri.writerow([])
        wri.writerow(["chi2", float(chi2)])
        wri.writerow(["chi2_red", float(chi2_red)])
        wri.writerow(["dof", int(dof)])
    files.append(par_csv)

    # Residus
    resid_csv = outdir / f"{prefix}_residus.csv"
    y_model = lorentz_amp(x_vals, *popt)  # ací x_vals està en omega
    resid = y_vals - y_model
    with open(resid_csv, "w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        wri.writerow([x_label, "y", "model", "residu", "sigma_y", "sigma_x", "sigma_eff"])
        for i in range(len(x_vals)):
            wri.writerow([float(x_vals[i]), float(y_vals[i]), float(y_model[i]), float(resid[i]),
                          float(sy_vals[i]), float(sx_vals[i]), float(sigma_eff[i])])
    files.append(resid_csv)

    # Corba (+ banda)
    curve_csv = outdir / f"{prefix}_corba_fit.csv"
    with open(curve_csv, "w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        if lo is None or hi is None:
            wri.writerow([x_label, "y_fit"])
            for xv, yv in zip(x_grid, y_grid):
                wri.writerow([float(xv), float(yv)])
        else:
            wri.writerow([x_label, "y_fit", "y_lo_68", "y_hi_68"])
            for xv, yv, l, h in zip(x_grid, y_grid, lo, hi):
                wri.writerow([float(xv), float(yv), float(l), float(h)])
    files.append(curve_csv)

    # Covariància
    cov_csv = outdir / f"{prefix}_covariancia.csv"
    with open(cov_csv, "w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        wri.writerow([""] + PARAM_NAMES)
        pc = np.array(pcov, dtype=float)
        for i, row in enumerate(pc):
            wri.writerow([PARAM_NAMES[i]] + [float(v) for v in row])
    files.append(cov_csv)

    # ZIP
    zip_path = outdir / f"{prefix}_resultats.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in files:
            zf.write(fp, arcname=fp.name)
    files.append(zip_path)

    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV amb columnes x,y(,sigma_x,sigma_y)")
    ap.add_argument("--outdir", default="resultats_lorentz", help="Carpeta d'eixida (default: resultats_lorentz)")
    ap.add_argument("--prefix", default="kundt", help="Prefix dels fitxers (default: kundt)")
    ap.add_argument("--x-col", default="x", help="Nom de la columna X (default: x)")
    ap.add_argument("--y-col", default="y", help="Nom de la columna Y (default: y)")
    ap.add_argument("--sx-col", default="sigma_x", help="Nom de la columna sigma_x (default: sigma_x)")
    ap.add_argument("--sy-col", default="sigma_y", help="Nom de la columna sigma_y (default: sigma_y)")
    ap.add_argument("--x-is-omega", action="store_true", help="Interpreta X com ω (rad/s) en compte d'Hz")
    ap.add_argument("--no-band", action="store_true", help="No pintar banda d'incertesa")
    ap.add_argument("--no-show", action="store_true", help="No mostrar figures (només guardar)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    if args.x_col not in df.columns or args.y_col not in df.columns:
        raise ValueError(f"El CSV ha de contindre columnes '{args.x_col}' i '{args.y_col}'.")

    x = df[args.x_col].to_numpy(float)
    y = df[args.y_col].to_numpy(float)

    # Errors: si no hi són, assumim 0 en x i 1 en y (ajust no ponderat de veritat)
    sx = df[args.sx_col].to_numpy(float) if args.sx_col in df.columns else np.zeros_like(x)
    sy = df[args.sy_col].to_numpy(float) if args.sy_col in df.columns else np.ones_like(y)

    idx = np.argsort(x)
    x, y, sx, sy = x[idx], y[idx], sx[idx], sy[idx]

    if args.x_is_omega:
        omega = x.copy()
        sigma_omega = sx.copy()
        x_plot = x
        x_grid_plot = np.linspace(float(x_plot.min()), float(x_plot.max()), 2500)
        omega_grid = x_grid_plot
        xlabel = "ω (rad/s)"
    else:
        omega = 2.0 * np.pi * x
        sigma_omega = 2.0 * np.pi * sx
        x_plot = x
        x_grid_plot = np.linspace(float(x_plot.min()), float(x_plot.max()), 2500)
        omega_grid = 2.0 * np.pi * x_grid_plot
        xlabel = "freqüència (Hz)"

    # Seeds i bounds
    p0 = estimate_seeds(omega, y)
    om_min, om_max = float(omega.min()), float(omega.max())
    bounds = ([0.0, om_min, 0.0], [np.inf, om_max, np.inf])

    popt, pcov, sigma_eff = fit_with_sigma_eff(omega, y, sy, sigma_omega, p0, bounds, n_iter=5)
    perr = np.sqrt(np.diag(pcov))

    y_model = lorentz_amp(omega, *popt)
    resid = y - y_model
    chi2 = float(np.sum((resid / sigma_eff) ** 2))
    dof = int(len(y) - len(popt))
    chi2_red = chi2 / max(dof, 1)

    print("\nResultats del fit (Lorentz)")
    print("chi2     =", chi2)
    print("chi2_red =", chi2_red)
    print("dof      =", dof)
    for n, v, e in zip(PARAM_NAMES, popt, perr):
        print(f"{n:8s} = {v: .6g}  ± {e:.2g}")

    if not args.x_is_omega:
        fr = popt[1] / (2.0 * np.pi)
        sfr = perr[1] / (2.0 * np.pi)
        print(f"\nFreq. ressonància (Hz) = {fr:.6g} ± {sfr:.2g}")

    y_grid = lorentz_amp(omega_grid, *popt)

    lo = hi = None
    if not args.no_band:
        lo, hi = mc_band_constrained(omega_grid, popt, pcov, bounds, n_draws=1200, seed=12345)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Plot principal (dades + fit + banda dins del mateix plot)
    plt.figure(figsize=(9.0, 4.8))
    plt.errorbar(x_plot, y, yerr=sy, xerr=sx, fmt="o", capsize=2, label="Dades")
    plt.plot(x_grid_plot, y_grid, linewidth=2, label="Ajust lorentzià")
    if lo is not None and hi is not None:
        plt.fill_between(x_grid_plot, lo, hi, alpha=0.25, label="Banda ~68%")
    plt.xlabel(xlabel)
    plt.ylabel("Amplitud (Vpp)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{args.prefix}_fit.png", dpi=200)

    # Residus
    plt.figure(figsize=(9.0, 3.2))
    plt.axhline(0, linewidth=1)
    plt.errorbar(x_plot, resid, yerr=sy, xerr=sx, fmt="o", capsize=2)
    plt.xlabel(xlabel)
    plt.ylabel("residu (dades - model)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / f"{args.prefix}_residus.png", dpi=200)

    # Export (guardem omega a residus/covariància; i x_grid_plot a la corba)
    # Per simplicitat, al fitxer residus guardem omega (no Hz). Si vols Hz, canvia-ho.
    files = export_all(
        outdir=outdir,
        prefix=args.prefix,
        x_label="omega_rad_s",
        x_vals=omega,
        y_vals=y,
        sx_vals=sigma_omega,
        sy_vals=sy,
        sigma_eff=sigma_eff,
        popt=popt,
        perr=perr,
        pcov=pcov,
        chi2=chi2,
        chi2_red=chi2_red,
        dof=dof,
        x_grid=x_grid_plot,   # ací guardem l'eix tal com es mostra al plot (Hz o ω)
        y_grid=y_grid,
        lo=lo,
        hi=hi
    )

    print("\nEixida guardada en:", outdir.resolve())
    print("Fitxers creats:")
    for fp in files:
        print(" -", fp.name)
    print("Figures PNG creades:")
    print(" -", f"{args.prefix}_fit.png")
    print(" -", f"{args.prefix}_residus.png")

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
