
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LagCorrConfig:
    split_col: str = "split"
    split_value: str = "train"
    station_col: str = "id_sonde"
    ts_col: str = "ts"
    y_col: str = "temp_water_c"
    exo_cols: Tuple[str, ...] = ("temp_air_eobs_c", "discharge_q", "rainf_eobs")
    max_lag_days: int = 7
    step_hours: int = 2
    min_n: int = 50


def _assert_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Colonnes manquantes: {missing}")


def lag_corr_values(y, x, max_lag_steps, min_n=50):
    """
    Calcule corr(y(t), x(t-lag)) pour lag = 0..max_lag_steps (causal).
    Retourne un array de taille max_lag_steps+1
    """
    vals: List[float] = []
    for lag in range(max_lag_steps + 1):
        tmp = pd.concat([y, x.shift(lag)], axis=1).dropna()
        vals.append(tmp.iloc[:, 0].corr(tmp.iloc[:, 1]) if len(tmp) >= min_n else np.nan)
    return np.asarray(vals, dtype=float)


def best_lag_from_corr(corr_vals):
    """Renvoie (best_lag_steps, best_corr) avec max |corr|. None si tout NaN."""
    if corr_vals.size == 0 or np.all(np.isnan(corr_vals)):
        return None, np.nan
    i = int(np.nanargmax(np.abs(corr_vals)))
    return i, float(corr_vals[i])


def compute_bestlags(df, config=LagCorrConfig(), return_curves=False):
    """
    Calcule le meilleur lag (max |corr|) par (station, exo) sur le split donné.
    Retour:
      - bestlags: long format (id_sonde, exo, best_lag_hours, best_corr)
      - pivot: pivot lisible (index=station, colonnes=exo, values=[best_lag_hours, best_corr])
      - curves (optionnel): long format des corrélations pour tous lags (station, exo, lag_hours, corr)
    """
    _assert_columns(df, [config.split_col, config.station_col, config.ts_col, config.y_col])
    _assert_columns(df, list(config.exo_cols))

    if not pd.api.types.is_datetime64_any_dtype(df[config.ts_col]):
        raise TypeError(f"{config.ts_col} doit être datetime64 (parse ts avant).")

    df_split = df.loc[df[config.split_col] == config.split_value].copy()
    if df_split.empty:
        raise ValueError(f"Aucune ligne pour {config.split_col} == {config.split_value}")


    max_lag_steps = int(config.max_lag_days * 24 / config.step_hours)
    step_hours = int(config.step_hours)

    rows = []
    curve_rows = [] if return_curves else None

    stations = sorted(df_split[config.station_col].dropna().unique().tolist())

    for sid in stations:
        g = (df_split[df_split[config.station_col] == sid]
             .sort_values(config.ts_col)
             .set_index(config.ts_col))

        y = g[config.y_col].dropna()

        for exo in config.exo_cols:
            corr_vals = lag_corr_values(y, g[exo], max_lag_steps=max_lag_steps, min_n=config.min_n)
            best_steps, best_corr = best_lag_from_corr(corr_vals)

            rows.append({
                config.station_col: sid,
                "exo": exo,
                "best_lag_hours": None if best_steps is None else best_steps * step_hours,
                "best_corr": best_corr,
            })

            if return_curves:
                lag_hours = np.arange(0, max_lag_steps + 1) * step_hours
                for lh, cv in zip(lag_hours, corr_vals):
                    curve_rows.append({
                        config.station_col: sid,
                        "exo": exo,
                        "lag_hours": int(lh),
                        "corr": float(cv) if not np.isnan(cv) else np.nan,
                    })

    bestlags = pd.DataFrame(rows)
    pivot = bestlags.pivot(index=config.station_col, columns="exo", values=["best_lag_hours", "best_corr"])

    curves_df = pd.DataFrame(curve_rows) if return_curves else None
    return bestlags, pivot, curves_df


def export_bestlags(bestlags, pivot, out_dir, prefix='lagcorr_bestlags_train'):
    """Export simple CSV (long + pivot)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bestlags.to_csv(out_dir / f"{prefix}.csv", index=False)
    pivot.to_csv(out_dir / f"{prefix}_pivot.csv")
