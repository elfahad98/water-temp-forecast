from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Iterable

import pandas as pd


def log_models_to_mlflow(
    *,
    use_mlflow,
    run_name,
    expected,
    horizons,
    seed,
    tables,
    artifacts=(),
    figures=(),
    tags=None,
    extra_params=None,):
    """
    Enregistre dans un seul run MLflow :
      - les paramètres globaux (expected, horizons, seed et extras)
      - les métriques par table (colonnes : mode, horizon, mae, rmse, smape, r2, n_preds|n)
      - les artefacts et figures s'ils existent

    `tables` : dictionnaire {préfixe: dataframe}
      ex. : {"ETS": ets_global, "RIDGE": ridge_results, ...}
    """
    if not use_mlflow:
        print("MLflow disabled (use_mlflow=False)")
        return

    try:
        import mlflow
    except Exception as e:
        print("[MLflow] import failed:", str(e)[:140])
        return

    def _log_metrics_table(df, prefix):
        if df is None or df.empty:
            return
        d = df.copy()


        if "n_preds" not in d.columns and "n" in d.columns:
            d["n_preds"] = d["n"]


        if "mode" not in d.columns:
            d["mode"] = "na"

        for _, rr in d.iterrows():
            mode = str(rr.get("mode", "na"))
            horizon = str(rr.get("horizon", ""))

            for k in ["mae", "rmse", "smape", "r2", "n_preds"]:
                if k in rr and pd.notna(rr[k]):
                    mlflow.log_metric(f"{prefix}_{k.upper()}_{mode}_{horizon}", float(rr[k]))

    def _log_artifact_if_exists(p):
        if p is None:
            return
        p = Path(p)
        if p.exists():
            mlflow.log_artifact(str(p))

    with mlflow.start_run(run_name=run_name):

        if tags:
            for k, v in tags.items():
                mlflow.set_tag(str(k), str(v))


        mlflow.log_param("expected_timedelta", expected)
        mlflow.log_param("horizons", horizons)
        mlflow.log_param("seed", int(seed))

        if extra_params:
            for k, v in extra_params.items():
                mlflow.log_param(str(k), str(v))


        for prefix, df in tables.items():
            if df is None:
                continue
            _log_metrics_table(df, prefix=prefix)


        for p in artifacts:
            _log_artifact_if_exists(Path(p))
        for p in figures:
            _log_artifact_if_exists(Path(p))

    print("MLflow: all models logged")
