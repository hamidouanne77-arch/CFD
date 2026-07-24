#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CHORD = 0.4200


def numeric_file(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "%", "!", "TITLE", "VARIABLES")):
            continue
        vals = []
        for token in re.split(r"[\s,]+", line):
            try:
                vals.append(float(token))
            except ValueError:
                pass
        if len(vals) >= 2:
            rows.append(vals)
    if not rows:
        raise ValueError(f"No numeric rows in {path}")
    ncol = min(len(r) for r in rows)
    return pd.DataFrame([r[:ncol] for r in rows])


def read_csv_loose(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=r"\s+", engine="python")


def find_col(df: pd.DataFrame, patterns: list[str]) -> str:
    normalized = {c: re.sub(r"[^a-z0-9]", "", str(c).lower()) for c in df.columns}
    for pat in patterns:
        p = re.sub(r"[^a-z0-9]", "", pat.lower())
        for col, norm in normalized.items():
            if p in norm:
                return col
    raise KeyError(f"No column matching {patterns}; available={list(df.columns)}")


def zero_crossings(x: np.ndarray, y: np.ndarray) -> list[tuple[float, int]]:
    out = []
    for i in range(len(x)-1):
        if not (np.isfinite(y[i]) and np.isfinite(y[i+1])):
            continue
        if y[i] == 0:
            out.append((float(x[i]), 0))
        elif y[i] * y[i+1] < 0:
            xc = x[i] - y[i] * (x[i+1]-x[i]) / (y[i+1]-y[i])
            direction = 1 if y[i] < y[i+1] else -1
            out.append((float(xc), direction))
    return out


def separation_pair(x: np.ndarray, cf: np.ndarray) -> tuple[float | None, float | None]:
    crossings = zero_crossings(x, cf)
    sep = next((xc for xc, d in crossings if d < 0 and xc > 0.4), None)
    reatt = None
    if sep is not None:
        reatt = next((xc for xc, d in crossings if d > 0 and xc > sep), None)
    return sep, reatt


def metrics(sim: pd.DataFrame, exp_cp: pd.DataFrame, exp_cf: pd.DataFrame) -> dict:
    xcol = find_col(sim, ["coordinate_x", "points0", "x"])
    cpcol = find_col(sim, ["pressure_coefficient", "pressurecoefficient", "cp"])
    cfcol = find_col(sim, ["skin_friction_coefficient_x", "skinfrictioncoefficientx", "cf_x", "skinfriction"])
    ypluscol = None
    try:
        ypluscol = find_col(sim, ["y_plus", "yplus"])
    except KeyError:
        pass

    wall = pd.DataFrame({
        "x_over_c": pd.to_numeric(sim[xcol], errors="coerce") / CHORD,
        "cp": pd.to_numeric(sim[cpcol], errors="coerce"),
        "cf": pd.to_numeric(sim[cfcol], errors="coerce"),
    }).dropna().sort_values("x_over_c")
    if ypluscol:
        wall["yplus"] = pd.to_numeric(sim.loc[wall.index, ypluscol], errors="coerce")
    wall = wall.groupby(wall["x_over_c"].round(10), as_index=False).mean(numeric_only=True)

    cp_exp = exp_cp.iloc[:, :2].copy(); cp_exp.columns = ["x_over_c", "cp"]
    cf_exp = exp_cf.iloc[:, :2].copy(); cf_exp.columns = ["x_over_c", "cf"]
    cp_exp = cp_exp.sort_values("x_over_c")
    cf_exp = cf_exp.sort_values("x_over_c")

    def rmse(exp: pd.DataFrame, field: str) -> tuple[float, float]:
        mask = (exp["x_over_c"] >= wall["x_over_c"].min()) & (exp["x_over_c"] <= wall["x_over_c"].max())
        sub = exp[mask]
        pred = np.interp(sub["x_over_c"], wall["x_over_c"], wall[field])
        err = pred - sub[field].to_numpy()
        return float(np.sqrt(np.mean(err**2))), float(np.mean(np.abs(err)))

    cp_rmse, cp_mae = rmse(cp_exp, "cp")
    cf_rmse, cf_mae = rmse(cf_exp, "cf")
    sep_sim, reatt_sim = separation_pair(wall["x_over_c"].to_numpy(), wall["cf"].to_numpy())
    sep_exp, reatt_exp = separation_pair(cf_exp["x_over_c"].to_numpy(), cf_exp["cf"].to_numpy())

    result = {
        "cp_rmse": cp_rmse,
        "cp_mae": cp_mae,
        "cf_rmse": cf_rmse,
        "cf_mae": cf_mae,
        "separation_xc_sim": sep_sim,
        "reattachment_xc_sim": reatt_sim,
        "separation_xc_exp_from_cf_zero_crossing": sep_exp,
        "reattachment_xc_exp_from_cf_zero_crossing": reatt_exp,
        "separation_error_c": None if sep_sim is None or sep_exp is None else sep_sim-sep_exp,
        "reattachment_error_c": None if reatt_sim is None or reatt_exp is None else reatt_sim-reatt_exp,
        "min_cf": float(wall["cf"].min()),
        "max_yplus": float(wall["yplus"].max()) if "yplus" in wall else None,
        "mean_yplus": float(wall["yplus"].mean()) if "yplus" in wall else None,
        "surface_points": int(len(wall)),
    }
    return result, wall, cp_exp, cf_exp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--cp-exp", type=Path, required=True)
    ap.add_argument("--cf-exp", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    exp_cp = numeric_file(args.cp_exp)
    exp_cf = numeric_file(args.cf_exp)
    summaries = []

    for case_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        candidates = list(case_dir.glob("surface_flow*.csv")) + list(case_dir.glob("surface*.csv"))
        if not candidates:
            continue
        surface = read_csv_loose(candidates[0])
        result, wall, cp_e, cf_e = metrics(surface, exp_cp, exp_cf)
        result["case"] = case_dir.name
        summaries.append(result)
        wall.to_csv(args.output / f"{case_dir.name}_wall.csv", index=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(cp_e["x_over_c"], cp_e["cp"], "o", ms=3, label="Expérience")
        ax.plot(wall["x_over_c"], wall["cp"], label="SU2")
        ax.set_xlabel("x/c"); ax.set_ylabel("Cp"); ax.set_title(f"{case_dir.name} — Cp")
        ax.grid(True); ax.legend(); fig.tight_layout(); fig.savefig(args.output / f"{case_dir.name}_cp.png", dpi=180); plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(cf_e["x_over_c"], cf_e["cf"], "o", ms=3, label="Expérience")
        ax.plot(wall["x_over_c"], wall["cf"], label="SU2")
        ax.axhline(0.0, lw=1)
        ax.set_xlabel("x/c"); ax.set_ylabel("Cf"); ax.set_title(f"{case_dir.name} — Cf et séparation")
        ax.grid(True); ax.legend(); fig.tight_layout(); fig.savefig(args.output / f"{case_dir.name}_cf.png", dpi=180); plt.close(fig)

    if not summaries:
        raise RuntimeError("No SU2 surface CSV found")
    summary = pd.DataFrame(summaries).sort_values("case")
    summary.to_csv(args.output / "validation_summary.csv", index=False)
    (args.output / "validation_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10, 6))
    valid = summary.dropna(subset=["reattachment_xc_sim"])
    ax.scatter(valid["case"], valid["reattachment_xc_sim"], label="SU2")
    if len(valid):
        ax.axhline(valid["reattachment_xc_exp_from_cf_zero_crossing"].iloc[0], ls="--", label="Expérience")
    ax.tick_params(axis="x", rotation=45); ax.set_ylabel("x_r/c"); ax.set_title("Convergence du rattachement")
    ax.grid(True); ax.legend(); fig.tight_layout(); fig.savefig(args.output / "reattachment_convergence.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
