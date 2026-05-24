from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Tuple

import pandas as pd


def _save_df(df, path, name):
    if df is None:
        print(f"[SKIP] {name}: df is None")
        return
    if isinstance(df, pd.DataFrame) and df.empty:
        print(f"[SKIP] {name}: empty dataframe")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print("Saved:", path)


def export_model_artifacts(
    *,
    baseline_best,
    ets_global,
    ets_station,
    preds_ets,
    ets_logs,
    arima_global,
    sarima_global,
    sarimax_global,
    ridge_metrics,
    ridge_preds,
    art_tbl_dir,
    art_met_dir,
    art_prd_dir,
    art_sum_dir,
):
    """
    Exporte tous les artefacts CSV à partir des tables déjà calculées.
    Retourne :
      - la concaténation de all_metrics si possible
      - un dictionnaire {nom: chemin} de tous les fichiers écrits
    """
    written: Dict[str, Path] = {}


    if arima_global is not None and not arima_global.empty:
        arima_global = arima_global.copy()
        arima_global["model"] = "arima"

    if sarima_global is not None and not sarima_global.empty:
        sarima_global = sarima_global.copy()
        sarima_global["model"] = "sarima"


    p = art_tbl_dir / "baseline_best_by_mode_horizon.csv"
    _save_df(baseline_best, p, "baseline_best")
    if baseline_best is not None and not baseline_best.empty:
        written["baseline_best"] = p


    p = art_met_dir / "ets_metrics_global.csv"
    _save_df(ets_global, p, "ets_global")
    if ets_global is not None and not ets_global.empty:
        written["ets_global"] = p

    p = art_met_dir / "ets_metrics_by_station.csv"
    _save_df(ets_station, p, "ets_station")
    if ets_station is not None and not ets_station.empty:
        written["ets_station"] = p

    p = art_prd_dir / "ets_predictions.csv"
    _save_df(preds_ets, p, "preds_ets")
    if preds_ets is not None and not preds_ets.empty:
        written["preds_ets"] = p

    p = art_sum_dir / "ets_fit_logs.csv"
    _save_df(ets_logs, p, "ets_logs")
    if ets_logs is not None and not ets_logs.empty:
        written["ets_logs"] = p


    p = art_met_dir / "arima_metrics_dev.csv"
    _save_df(arima_global, p, "arima_global")
    if arima_global is not None and not arima_global.empty:
        written["arima_global"] = p

    p = art_met_dir / "sarima_metrics_dev.csv"
    _save_df(sarima_global, p, "sarima_global")
    if sarima_global is not None and not sarima_global.empty:
        written["sarima_global"] = p


    p = art_met_dir / "sarimax_metrics_dev.csv"
    _save_df(sarimax_global, p, "sarimax_global")
    if sarimax_global is not None and not sarimax_global.empty:
        written["sarimax_global"] = p


    p = art_met_dir / "ridge_metrics_val.csv"
    _save_df(ridge_metrics, p, "ridge_metrics")
    if ridge_metrics is not None and not ridge_metrics.empty:
        written["ridge_metrics"] = p

    p = art_prd_dir / "ridge_predictions_val.csv"
    _save_df(ridge_preds, p, "ridge_preds")
    if ridge_preds is not None and not ridge_preds.empty:
        written["ridge_preds"] = p


    dfs = []
    for df, tag in [
        (baseline_best, "baseline_best"),
        (ets_global, "ets"),
        (arima_global, "arima"),
        (sarima_global, "sarima"),
        (sarimax_global, "sarimax"),
        (ridge_metrics, "ridge"),
    ]:
        if df is None or df.empty:
            continue
        d = df.copy()
        if "model" not in d.columns:
            d["model"] = tag
        dfs.append(d)

    all_metrics = None
    if dfs:
        all_metrics = pd.concat(dfs, ignore_index=True)
        p = art_met_dir / "all_models_metrics_concat.csv"
        _save_df(all_metrics, p, "all_metrics")
        written["all_metrics"] = p
    else:
        print("[WARN] No metrics tables found to concat.")

    return all_metrics, written
