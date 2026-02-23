#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pràctica 7 — Mesura de G: ajust no lineal de P(t) + plots + export

Ús:
  python practicaG.py mesuresG.csv

CSV amb columnes: x, y, sy, sx
  x  : temps (s)
  y  : posició P (unitats del regle)
  sy : error en y
  sx : error en x

Eixida (automàtica):
- Guarda figures PNG: fitG_fit.png, fitG_residus.png (dins d'una carpeta d'eixida)
- Exporta fitxers: resultats_fit.txt/csv, residus.csv, corba_fit.csv, covariancia.csv, zip

____________________________________________________________-
Qué necessites:

Posa a la mateixa carpeta:
- ajust_mesuraG.py
- mesuresG.csv

1) Instal·la Miniconda (busca “Miniconda download”) amb les opcions per defecte.

2) Obri una terminal
- Windows: “Miniconda Prompt” o “Anaconda Prompt”
- macOS: Terminal
- Linux: Terminal

3) Crea un entorn i instal·la el que cal (una sola vegada)
Per fer açó, copia i pega:
conda create -n labmec python=3.11 numpy scipy pandas matplotlib -y
conda activate labmec

Ara, si en la terminal vas a la carpeta on estan els fitxers pots executar el script:

python practicaG.py mesuresG.csv
"""

import argparse
import csv
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -------------------------
# Model del sistema: P(t)
# -------------------------
def model_func(t, Pe, Sl, alpha, T, t_shift):
    """
    P(t)=Pe + Sl*[1 - exp(-alpha*dt/2)*(cos(w1*dt) + alpha/(2*w1)*sin(w1*dt))]
    w1 = 2*pi/T
    dt = t - t_shift
    """
    dt = t - t_shift
    w1 = 2 * np.pi / T
    expo = np.exp(-alpha * dt / 2.0)
    return Pe + Sl * (1.0 - expo * (np.cos(w1 * dt) + (alpha / (2 * w1)) * np.sin(w1 * dt)))


PARAM_NAMES = ["Pe", "Sl", "alpha", "T", "t_shift"]


def estimate_seeds(x, y):
    Pe0 = float(np.median(y[: min(5, len(y))]))
    Sl0 = float(np.median(y[-min(5, len(y)) :] ) - Pe0)

    # Estimació de T amb separació de màxims locals
    peaks = []
    for i in range(1, len(y) - 1):
        if (y[i] > y[i - 1]) and (y[i] >= y[i + 1]):
            peaks.append(i)

    if len(peaks) >= 2:
        T0 = float(np.median(np.diff(x[peaks])))
    else:
        T0 = float((x.max() - x.min()) / 4.0) if len(x) > 1 else 500.0

    alpha0 = 4e-4
    t_shift0 = float(x.min())
    return [Pe0, Sl0, alpha0, T0, t_shift0]


def dPdt_numeric(t, params, eps=1e-2):
    return (model_func(t + eps, *params) - model_func(t - eps, *params)) / (2 * eps)


def fit_with_sigma_eff(x, y, sy, sx, p0, bounds, n_iter=5):
    """
    Itera un fit ponderat amb sigma_eff:
      sigma_eff^2 = sy^2 + (dP/dt * sx)^2
    """
    sigma = sy.copy().astype(float)
    popt = None
    pcov = None

    for _ in range(int(n_iter)):
        popt, pcov = curve_fit(
            model_func,
            x,
            y,
            p0=(p0 if popt is None else popt),
            sigma=sigma,
            absolute_sigma=True,
            bounds=bounds,
            maxfev=250000,
        )
        dp = dPdt_numeric(x, popt, eps=1e-2)
        sigma = np.sqrt(sy**2 + (dp * sx) ** 2)

    return popt, pcov, sigma


# -------------------------
# Banda d'incertesa (MC covariància) amb restriccions
# -------------------------
def make_psd(cov, jitter=1e-15):
    cov = 0.5 * (cov + cov.T)
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 0.0, None)
    cov_psd = (V * w) @ V.T
    scale = float(np.max(np.diag(cov_psd))) if cov_psd.size else 1.0
    cov_psd = cov_psd + (jitter * (scale if scale > 0 else 1.0)) * np.eye(cov_psd.shape[0])
    return cov_psd


def mc_band_constrained(x_grid, popt, pcov, bounds, n_draws=1200, seed=12345):
    """
    Mostreja paràmetres ~N(popt, pcov) però:
    - imposa alpha >= 0 i T > 0
    - aplica bounds del fit
    - descarta simulacions amb NaN/inf
    Retorna (lo, hi) o (None, None) si no pot.
    """
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

        alpha = draws[:, 2]
        T = draws[:, 3]
        ok = (alpha >= 0.0) & (T > 0.0)
        ok = ok & np.all(draws >= lo_b, axis=1) & np.all(draws <= hi_b, axis=1)

        draws = draws[ok]
        if draws.size == 0:
            continue

        for p in draws:
            yv = model_func(x_grid, *p)
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


# -------------------------
# Export
# -------------------------
def export_all(outdir, prefix, x, y, sy, sx, sigma_eff, popt, perr, pcov, chi2, chi2_red, dof, x_grid, y_grid, lo, hi):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = []

    # TXT resum
    txt_path = outdir / f"{prefix}_resultats_fit.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Resultats del fit (Pràctica 7 — Mesura de G)\n")
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
    y_model = model_func(x, *popt)
    resid = y - y_model
    with open(resid_csv, "w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        wri.writerow(["t", "P", "model", "residu", "sy", "sx", "sigma_eff"])
        for i in range(len(x)):
            wri.writerow([float(x[i]), float(y[i]), float(y_model[i]), float(resid[i]),
                          float(sy[i]), float(sx[i]), float(sigma_eff[i])])
    files.append(resid_csv)

    # Corba + banda (si hi és)
    curve_csv = outdir / f"{prefix}_corba_fit.csv"
    with open(curve_csv, "w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        if lo is None or hi is None:
            wri.writerow(["t_grid", "P_fit"])
            for t_, p_ in zip(x_grid, y_grid):
                wri.writerow([float(t_), float(p_)])
        else:
            wri.writerow(["t_grid", "P_fit", "P_lo_68", "P_hi_68"])
            for t_, p_, l_, h_ in zip(x_grid, y_grid, lo, hi):
                wri.writerow([float(t_), float(p_), float(l_), float(h_)])
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
    ap.add_argument("csv", help="CSV amb columnes x,y,sy,sx")
    ap.add_argument("--outdir", default="resultats_fitG", help="Carpeta d'eixida (default: resultats_fitG)")
    ap.add_argument("--prefix", default="fitG", help="Prefix dels fitxers (default: fitG)")
    ap.add_argument("--no-band", action="store_true", help="No dibuixar / no exportar banda d'incertesa")
    ap.add_argument("--no-show", action="store_true", help="No mostrar figures (només guardar)")
    args = ap.parse_args()

    # Carrega dades
    df = pd.read_csv(args.csv)
    for col in ("x", "y", "sy", "sx"):
        if col not in df.columns:
            raise ValueError(f"Falta la columna '{col}' en {args.csv}. Necessites: x,y,sy,sx")

    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)
    sy = df["sy"].to_numpy(float)
    sx = df["sx"].to_numpy(float)

    idx = np.argsort(x)
    x, y, sy, sx = x[idx], y[idx], sy[idx], sx[idx]

    # Seeds i bounds
    p0 = estimate_seeds(x, y)
    ymin, ymax = float(y.min()), float(y.max())
    rng_y = (ymax - ymin) if (ymax > ymin) else 1.0

    bounds = (
        [ymin - 5 * rng_y, -10 * rng_y, 0.0,   10.0,   float(x.min() - 2000.0)],
        [ymax + 5 * rng_y,  10 * rng_y, 0.05, 2000.0,  float(x.min() + 2000.0)],
    )

    # Fit
    popt, pcov, sigma_eff = fit_with_sigma_eff(x, y, sy, sx, p0, bounds, n_iter=5)
    perr = np.sqrt(np.diag(pcov))

    resid = y - model_func(x, *popt)
    chi2 = float(np.sum((resid / sigma_eff) ** 2))
    dof = int(len(x) - len(popt))
    chi2_red = chi2 / max(dof, 1)

    print("\nResultats del fit")
    print("chi2     =", chi2)
    print("chi2_red =", chi2_red)
    print("dof      =", dof)
    print("")
    for n, v, e in zip(PARAM_NAMES, popt, perr):
        print(f"{n:8s} = {v: .6g}  ± {e:.2g}")

    # Corba
    x_grid = np.linspace(float(x.min()), float(x.max()), 2500)
    y_grid = model_func(x_grid, *popt)

    lo = hi = None
    if not args.no_band:
        lo, hi = mc_band_constrained(x_grid, popt, pcov, bounds, n_draws=1200, seed=12345)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Figura 1: dades + fit (+ banda si hi és)
    plt.figure(figsize=(9.0, 4.8))
    plt.errorbar(x, y, yerr=sy, xerr=sx, fmt="o", capsize=2, label="Dades")
    plt.plot(x_grid, y_grid, linewidth=2, label="Model ajustat")
    if lo is not None and hi is not None:
        plt.fill_between(x_grid, lo, hi, alpha=0.25, label="Banda ~68% (paràmetres)")
    plt.xlabel("t (s)")
    plt.ylabel("P (unitats del regle)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{args.prefix}_fit.png", dpi=200)

    # Figura 2: residus
    plt.figure(figsize=(9.0, 3.2))
    plt.axhline(0, linewidth=1)
    plt.errorbar(x, resid, yerr=sy, xerr=sx, fmt="o", capsize=2)
    plt.xlabel("t (s)")
    plt.ylabel("residu (P - model)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / f"{args.prefix}_residus.png", dpi=200)

    # Exporta fitxers
    files = export_all(
        outdir=outdir,
        prefix=args.prefix,
        x=x, y=y, sy=sy, sx=sx, sigma_eff=sigma_eff,
        popt=popt, perr=perr, pcov=pcov,
        chi2=chi2, chi2_red=chi2_red, dof=dof,
        x_grid=x_grid, y_grid=y_grid, lo=lo, hi=hi
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
