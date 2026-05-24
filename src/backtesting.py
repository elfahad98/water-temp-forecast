
from __future__ import annotations

from typing import Mapping
import numpy as np
import pandas as pd


def _to_dt_index(idx):
    """Convertit en DatetimeIndex trié + unique."""
    return pd.DatetimeIndex(pd.to_datetime(pd.Index(idx).unique())).sort_values()


def _nearest_on_index(t, idx):
    """
    Ramène t au timestamp précédent existant dans idx.
    Utile pour aligner les bornes sur l'index réel.
    """
    pos = idx.searchsorted(pd.Timestamp(t), side="right") - 1
    pos = max(0, min(pos, len(idx) - 1))
    return pd.Timestamp(idx[pos])


def make_folds(
    idx,
    bt_start,
    bt_end,
    *,
    step=pd.Timedelta(days=30),
    eval_window=pd.Timedelta(days=30),
    min_train=pd.Timedelta(days=365),
    mode='expanding',
    rolling_window=pd.Timedelta(days=730),
    expected=pd.Timedelta(hours=2),
):
    """
    Construit des folds de backtesting sur un index temps régulier.

    Chaque fold contient :
      - train_start, train_end
      - eval_start, eval_end
      - mode (expanding / rolling)

    Convention :
      - eval_start = train_end + expected (2h)
      - eval_end   = train_end + eval_window (aligné sur idx)
    """
    idx = _to_dt_index(idx)

    if len(idx) < 2:
        raise ValueError("Index temporel trop court pour construire des folds.")


    if (idx[1] - idx[0]) != expected:
        raise AssertionError(
            f"Index temps non régulier ({idx[1] - idx[0]}), attendu {expected}."
        )

    bt_start = pd.Timestamp(bt_start)
    bt_end = pd.Timestamp(bt_end)

    if mode not in {"expanding", "rolling"}:
        raise ValueError("mode doit être 'expanding' ou 'rolling'.")

    folds = []
    last_train_end = bt_end - eval_window
    train_end = bt_start + min_train
    fold_id = 0

    while train_end <= last_train_end:
        train_end_aligned = _nearest_on_index(train_end, idx)

        if mode == "expanding":
            train_start_aligned = _nearest_on_index(bt_start, idx)
        else:
            train_start_aligned = _nearest_on_index(train_end_aligned - rolling_window + expected, idx)
            if train_start_aligned < bt_start:
                train_start_aligned = _nearest_on_index(bt_start, idx)

        eval_start_aligned = _nearest_on_index(train_end_aligned + expected, idx)
        eval_end_aligned = _nearest_on_index(train_end_aligned + eval_window, idx)


        if (
            train_start_aligned < train_end_aligned
            and eval_start_aligned <= eval_end_aligned
            and eval_end_aligned <= bt_end
        ):
            fold_id += 1
            folds.append(
                {
                    "fold_id": fold_id,
                    "mode": mode,
                    "train_start": train_start_aligned,
                    "train_end": train_end_aligned,
                    "eval_start": eval_start_aligned,
                    "eval_end": eval_end_aligned,
                }
            )

        train_end = train_end + step

    return pd.DataFrame(folds)


def build_series_by_station(df, *, id_col='id_sonde', ts_col='ts', target_col='temp_water_c'):
    """
    Transforme le dataframe en dict {station: série temporelle indexée par ts}
    """
    needed = {id_col, ts_col, target_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans df: {missing}")

    out: dict[int, pd.Series] = {}
    for sid, g in (
        df[[id_col, ts_col, target_col]]
        .dropna(subset=[id_col, ts_col])
        .groupby(id_col, sort=True)
    ):
        gg = g.sort_values(ts_col)
        out[int(sid)] = gg.set_index(ts_col)[target_col]
    return out


def smape(y_true, y_pred, eps=1e-08):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(200.0 * np.mean(np.abs(y_true - y_pred) / denom))


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "smape": smape(y_true, y_pred),}


def evaluate_baselines_walkforward(
    folds,
    series_by_station,
    *,
    horizons,
    season_by_h=None,
    expected=pd.Timedelta(hours=2),
    mode_name=None,
):
    """
    Évalue des baselines en avance pas à pas sur tous les timestamps
    de chaque fenêtre d'évaluation.

    Pour chaque timestamp cible t :
      - naive(h) :    y_hat(t) = y(t-h)
      - seasonal(m) : y_hat(t) = y(t-m)

    Retourne :
      - preds (toutes les prédictions)
      - metrics_global (agrégées par mode, modèle et horizon)
      - metrics_station (agrégées par station)
    """
    if season_by_h is None:
        season_by_h = {"h2": 12, "d1": 12, "w1": 84}

    required_cols = {"fold_id", "train_end", "eval_start", "eval_end"}
    if not required_cols.issubset(folds.columns):
        raise ValueError(f"folds doit contenir au minimum {required_cols}")

    rows = []

    for _, r in folds.iterrows():
        fold_id = int(r["fold_id"])
        row_mode = mode_name or str(r.get("mode", ""))

        train_end = pd.Timestamp(r["train_end"])
        eval_start = pd.Timestamp(r["eval_start"])
        eval_end = pd.Timestamp(r["eval_end"])

        for sid, y in series_by_station.items():
            y = y.sort_index()


            eval_targets = y.index[(y.index >= eval_start) & (y.index <= eval_end)]
            if len(eval_targets) == 0:
                continue

            for h_name, h_steps in horizons.items():
                h_delta = int(h_steps) * expected

                if h_name not in season_by_h:
                    raise ValueError(f"horizon '{h_name}' absent de season_by_h")
                m = int(season_by_h[h_name])
                m_delta = m * expected

                for target_ts in eval_targets:
                    y_true = y.get(target_ts, np.nan)
                    if pd.isna(y_true):
                        continue


                    src_naive_ts = target_ts - h_delta
                    if src_naive_ts in y.index:
                        rows.append(
                            {
                                "mode": row_mode,
                                "fold_id": fold_id,
                                "id_sonde": int(sid),
                                "train_end": train_end,
                                "eval_start": eval_start,
                                "eval_end": eval_end,
                                "target_ts": pd.Timestamp(target_ts),
                                "horizon": h_name,
                                "model": "naive",
                                "y_true": float(y_true),
                                "y_pred": float(y.loc[src_naive_ts]),
                                "source_ts": pd.Timestamp(src_naive_ts),
                            }
                        )


                    src_season_ts = target_ts - m_delta
                    if src_season_ts in y.index:
                        rows.append(
                            {
                                "mode": row_mode,
                                "fold_id": fold_id,
                                "id_sonde": int(sid),
                                "train_end": train_end,
                                "eval_start": eval_start,
                                "eval_end": eval_end,
                                "target_ts": pd.Timestamp(target_ts),
                                "horizon": h_name,
                                "model": f"seasonal_m{m}",
                                "y_true": float(y_true),
                                "y_pred": float(y.loc[src_season_ts]),
                                "source_ts": pd.Timestamp(src_season_ts),
                            }
                        )

    preds = pd.DataFrame(rows)
    if preds.empty:
        raise ValueError("Aucune prédiction baseline générée.")


    metrics_global = (
        preds.groupby(["mode", "model", "horizon"], sort=True)[["y_true", "y_pred"]]
        .apply(lambda g: pd.Series(compute_metrics(g["y_true"], g["y_pred"])))
        .reset_index()
        .sort_values(["mode", "horizon", "model"])
        .reset_index(drop=True))


    metrics_station = (
        preds.groupby(["mode", "id_sonde", "model", "horizon"], sort=True)[["y_true", "y_pred"]]
        .apply(lambda g: pd.Series(compute_metrics(g["y_true"], g["y_pred"])))
        .reset_index()
        .sort_values(["mode", "id_sonde", "horizon", "model"])
        .reset_index(drop=True))


    counts = (
        preds.groupby(["mode", "model", "horizon"], sort=True)
        .size()
        .reset_index(name="n_preds")
        .sort_values(["mode", "horizon", "model"])
        .reset_index(drop=True))
    metrics_global = metrics_global.merge(
        counts, on=["mode", "model", "horizon"], how="left")

    return preds, metrics_global, metrics_station
