from __future__ import annotations

import numpy as np
import pandas as pd
import warnings

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

def smape_np(y_true, y_pred, eps=1e-08):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(200.0 * np.mean(np.abs(y_true - y_pred) / denom))


def eval_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    smape = smape_np(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "smape": float(smape),
        "r2": float(r2),}


def build_future_targets(
    df_model,
    *,
    id_col='id_sonde',
    ts_col='ts',
    target_col='temp_water_c',
    split_col='split',
    horizons,
):
    """
    Crée les cibles futures y_h = y(t+h) par station.
    Met y_h à NaN lorsque y(t+h) appartient à un split différent
    afin de préserver les frontières entre ensembles.
    """
    df_tmp = df_model[[id_col, ts_col, split_col, target_col]].copy()
    df_tmp = df_tmp.sort_values([id_col, ts_col]).reset_index(drop=True)

    for h_name, h_steps in horizons.items():
        h_steps = int(h_steps)


        df_tmp[f"split_target_{h_name}"] = df_tmp.groupby(id_col)[split_col].shift(-h_steps)


        df_tmp[f"y_{h_name}"] = df_tmp.groupby(id_col)[target_col].shift(-h_steps)


        mismatch = df_tmp[f"split_target_{h_name}"].ne(df_tmp[split_col])
        df_tmp.loc[mismatch, f"y_{h_name}"] = np.nan

    return df_tmp


def build_global_feature_matrix(
    df_model,
    *,
    id_col='id_sonde',
    ts_col='ts',
    target_col='temp_water_c',
    split_col='split',
    keep_non_numeric=False,
):
    """
    Construit X_all pour un modèle global :
    - encodage indicateur sur id_col
    - retrait des colonnes non numériques par défaut
    Retourne :
      - df_sorted
      - X_all
      - feat_cols
      - non_numeric_cols retirées
    """
    df_sorted = df_model.sort_values([id_col, ts_col]).reset_index(drop=True)

    non_feat = {id_col, ts_col, target_col, split_col}
    feat_cols = [c for c in df_sorted.columns if c not in non_feat]

    X_all = df_sorted[[id_col, split_col] + feat_cols].copy()
    X_all = pd.get_dummies(X_all, columns=[id_col], prefix="st", drop_first=False)

    non_numeric_cols: list[str] = []
    if not keep_non_numeric:
        non_numeric_cols = (
            X_all.drop(columns=[split_col])
            .select_dtypes(exclude=["number", "bool"])
            .columns
            .tolist()
        )
        if non_numeric_cols:
            X_all = X_all.drop(columns=non_numeric_cols)

    return df_sorted, X_all, feat_cols, non_numeric_cols


def make_xy_for_horizon(X_all, df_targets, *, h_name, split_col='split'):
    """
    Retourne X/y pour train, val, test pour un horizon donné.
    Supprime les lignes avec NaN dans X ou dans y_future.
    """
    y_all = df_targets[f"y_{h_name}"]

    X_tr = X_all[X_all[split_col] == "train"].drop(columns=[split_col])
    y_tr = y_all[df_targets[split_col] == "train"]

    X_va = X_all[X_all[split_col] == "val"].drop(columns=[split_col])
    y_va = y_all[df_targets[split_col] == "val"]

    X_te = X_all[X_all[split_col] == "test"].drop(columns=[split_col])
    y_te = y_all[df_targets[split_col] == "test"]

    def _dropna_xy(Xp, yp):
        mask = Xp.notna().all(axis=1) & yp.notna()
        Xp2 = Xp.loc[mask].reset_index(drop=True)
        yp2 = yp.loc[mask].reset_index(drop=True)
        return Xp2, yp2

    X_tr, y_tr = _dropna_xy(X_tr, y_tr)
    X_va, y_va = _dropna_xy(X_va, y_va)
    X_te, y_te = _dropna_xy(X_te, y_te)

    return X_tr, y_tr, X_va, y_va, X_te, y_te


def fit_ridge_multihorizon(
    df_model,
    *,
    horizons,
    id_col='id_sonde',
    ts_col='ts',
    target_col='temp_water_c',
    split_col='split',
    alpha=1.0,
    seed=42,
):
    """
    Entraîne une Ridge par horizon et évalue sur VAL.
    Retourne :
      - results_df
      - preds_val_df
      - models_by_h
      - meta
    """
    df_targets = build_future_targets(
        df_model,
        id_col=id_col,
        ts_col=ts_col,
        target_col=target_col,
        split_col=split_col,
        horizons=horizons,
    )

    df_sorted, X_all, feat_cols, non_numeric_cols = build_global_feature_matrix(
        df_model,
        id_col=id_col,
        ts_col=ts_col,
        target_col=target_col,
        split_col=split_col,
        keep_non_numeric=False,
    )


    assert (df_sorted[id_col].values == df_targets[id_col].values).all()
    assert (df_sorted[ts_col].values == df_targets[ts_col].values).all()

    results = []
    preds_val_rows = []
    models_by_h: dict[str, Pipeline] = {}

    for h_name in horizons.keys():
        Xtr, ytr, Xva, yva, Xte, yte = make_xy_for_horizon(
            X_all, df_targets, h_name=h_name, split_col=split_col
        )

        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("model", Ridge(alpha=alpha, random_state=seed)),
        ])

        pipe.fit(Xtr, ytr)
        pred_va = pipe.predict(Xva)

        mets = eval_metrics(yva, pred_va)
        results.append({
            "model": "ridge",
            "horizon": h_name,
            "split": "val",
            "mae": mets["mae"],
            "rmse": mets["rmse"],
            "smape": mets["smape"],
            "r2": mets["r2"],
            "n": int(len(yva)),
        })


        idx_val = df_targets[df_targets[split_col] == "val"].index
        Xva_mask = X_all[X_all[split_col] == "val"].drop(columns=[split_col]).notna().all(axis=1)
        yva_mask = df_targets.loc[idx_val, f"y_{h_name}"].notna().values
        keep_mask = Xva_mask.values & yva_mask

        df_val_meta = df_targets.loc[idx_val, [id_col, ts_col, split_col]].reset_index(drop=True)
        df_val_meta = df_val_meta.loc[keep_mask].reset_index(drop=True)

        tmp_pred = df_val_meta.copy()
        tmp_pred["horizon"] = h_name
        tmp_pred["y_true"] = yva.reset_index(drop=True)
        tmp_pred["y_pred"] = pd.Series(pred_va).reset_index(drop=True)
        preds_val_rows.append(tmp_pred)

        models_by_h[h_name] = pipe

    results_df = pd.DataFrame(results).sort_values(["horizon"]).reset_index(drop=True)
    preds_val_df = pd.concat(preds_val_rows, ignore_index=True)

    meta = {
        "feat_cols_raw": feat_cols,
        "non_numeric_cols_removed": non_numeric_cols,
        "x_all_shape": tuple(X_all.shape),
        "n_features_final": int(X_all.drop(columns=[split_col]).shape[1]),
    }

    return results_df, preds_val_df, models_by_h, meta

def build_series_by_station(df, *, id_col='id_sonde', ts_col='ts', target_col='temp_water_c'):
    """
    Retourne un dict {station: série temporelle indexée par ts}.
    """
    series_by_station = {
        int(sid): g.sort_values(ts_col).set_index(ts_col)[target_col].astype(float)
        for sid, g in df[[id_col, ts_col, target_col]].groupby(id_col, sort=True)
    }
    return series_by_station


def fit_ets_with_fallback(
    y_train,
    *,
    seasonal_periods=12,
    optimizer_method='L-BFGS-B',
    optimizer_maxiter=200,
    init_method='estimated',
):
    """
    Essaie plusieurs configurations ETS de la plus riche à la plus simple.
    """
    attempts = [
        dict(trend="add", damped_trend=True, seasonal="add", seasonal_periods=seasonal_periods),
        dict(trend="add", damped_trend=True, seasonal=None, seasonal_periods=None),
        dict(trend=None, damped_trend=False, seasonal=None, seasonal_periods=None),]

    last_err = None
    for cfg in attempts:
        try:
            model = ExponentialSmoothing(
                y_train,
                trend=cfg["trend"],
                damped_trend=cfg["damped_trend"],
                seasonal=cfg["seasonal"],
                seasonal_periods=cfg["seasonal_periods"],
                initialization_method=init_method,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)

                try:
                    fit_res = model.fit(
                        optimized=True,
                        use_brute=False,
                        method=optimizer_method,
                        minimize_kwargs={"maxiter": int(optimizer_maxiter)},
                    )
                    fit_tag = f"{optimizer_method},maxiter={optimizer_maxiter}"
                except TypeError:
                    fit_res = model.fit(
                        optimized=True,
                        use_brute=False,
                    )
                    fit_tag = "default_optimizer"

            mle = getattr(fit_res, "mle_retvals", None)
            if isinstance(mle, dict) and (mle.get("converged") is False):
                raise RuntimeError(f"ETS did not converge: {mle}")

            cfg_name = (
                f"trend={cfg['trend']},"
                f"damped={cfg['damped_trend']},"
                f"seasonal={cfg['seasonal']},"
                f"sp={cfg['seasonal_periods']},"
                f"init={init_method},"
                f"method={optimizer_method},"
                f"maxiter={optimizer_maxiter}"
                f"opt={fit_tag}"
            )
            return fit_res, cfg_name

        except Exception as e:
            last_err = e

    raise RuntimeError(f"ETS fit failed after fallbacks: {last_err}")

def eval_ets_on_folds(
    folds,
    series_by_station,
    *,
    horizons,
    expected=pd.Timedelta(hours=2),
    mode_name,
    seasonal_periods=12,
    max_train_days=365,
    optimizer_method='L-BFGS-B',
    optimizer_maxiter=200,
    init_method='heuristic',
):
    """
    Évalue ETS avec 1 origine de prévision par fold / station.
    Retourne :
      - preds
      - metrics_global
      - metrics_station
      - fit_logs_df
    """
    rows = []
    fit_logs = []

    max_h = max(horizons.values())

    for _, r in folds.iterrows():
        fold_id = int(r["fold_id"])
        train_start = pd.Timestamp(r["train_start"])
        train_end = pd.Timestamp(r["train_end"])
        eval_end = pd.Timestamp(r["eval_end"])

        for sid, y in series_by_station.items():
            y = y.sort_index()

            train_start_eff = train_start
            if max_train_days is not None:
                train_start_eff = max(train_start, train_end - pd.Timedelta(days=int(max_train_days)))

            y_train = y[(y.index >= train_start_eff) & (y.index <= train_end)].copy()
            y_train = y_train.asfreq(expected)

            if len(y_train) < max(2 * seasonal_periods, 30):
                continue
            if y_train.isna().any():
                continue
            if train_end not in y_train.index:
                continue

            try:
                fit_res, used_cfg = fit_ets_with_fallback(
                    y_train,
                    seasonal_periods=seasonal_periods,
                    optimizer_method=optimizer_method,
                    optimizer_maxiter=optimizer_maxiter,
                    init_method=init_method,)

                fcst = fit_res.forecast(max_h)

                fit_logs.append({
                    "mode": mode_name,
                    "fold_id": fold_id,
                    "id_sonde": sid,
                    "train_end": train_end,
                    "config_used": used_cfg,
                    "n_train": int(len(y_train)),
                })

            except Exception as e:
                fit_logs.append({
                    "mode": mode_name,
                    "fold_id": fold_id,
                    "id_sonde": sid,
                    "train_end": train_end,
                    "config_used": f"FAILED: {e}",
                    "n_train": int(len(y_train)),
                })
                continue

            for h_name, h_steps in horizons.items():
                target_ts = train_end + h_steps * expected

                if target_ts > eval_end:
                    continue
                if target_ts not in y.index:
                    continue

                y_true = y.loc[target_ts]
                y_pred = float(fcst.iloc[h_steps - 1])

                rows.append({
                    "mode": mode_name,
                    "fold_id": fold_id,
                    "id_sonde": sid,
                    "train_start": train_start,
                    "train_end": train_end,
                    "target_ts": target_ts,
                    "horizon": h_name,
                    "model": "ets",
                    "y_true": float(y_true),
                    "y_pred": y_pred,
                })

    preds = pd.DataFrame(rows)
    fit_logs_df = pd.DataFrame(fit_logs)

    if preds.empty:
        fit_logs_df = pd.DataFrame(fit_logs)
        if not fit_logs_df.empty:
            print("Top failures / configs:")
            print(fit_logs_df["config_used"].value_counts().head(10))
        raise ValueError(f"Aucune prédiction ETS générée pour mode={mode_name}")

    metrics_global = (
        preds.groupby(["mode", "model", "horizon"], sort=True)[["y_true", "y_pred"]]
        .apply(lambda g: pd.Series(eval_metrics(g["y_true"], g["y_pred"])))
        .reset_index()
        .sort_values(["mode", "horizon", "model"])
        .reset_index(drop=True)
    )

    metrics_station = (
        preds.groupby(["mode", "id_sonde", "model", "horizon"], sort=True)[["y_true", "y_pred"]]
        .apply(lambda g: pd.Series(eval_metrics(g["y_true"], g["y_pred"])))
        .reset_index()
        .sort_values(["mode", "id_sonde", "horizon"])
        .reset_index(drop=True)
    )

    counts = (
        preds.groupby(["mode", "model", "horizon"], sort=True)
        .size()
        .reset_index(name="n_preds")
        .sort_values(["mode", "horizon"])
        .reset_index(drop=True)
    )

    metrics_global = metrics_global.merge(
        counts, on=["mode", "model", "horizon"], how="left"
    )

    return preds, metrics_global, metrics_station, fit_logs_df


def _fit_sarimax_try(y_train, *, order, seasonal_order, maxiter=80):
    """
    Ajuste un modèle SARIMAX robuste (univarié).
    Retourne l'objet de résultats estimé.
    """
    mod = SARIMAX(
        y_train,
        order=order,
        seasonal_order=seasonal_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = mod.fit(disp=False, maxiter=int(maxiter))

    return res


def eval_sarima_on_folds(
    folds,
    series_by_station,
    *,
    horizons,
    expected=pd.Timedelta(hours=2),
    mode_name,
    max_train_days=365,
    maxiter=80,
    min_train_points=200,
    candidates=None,
):
    """
    Évalue ARIMA/SARIMA (SARIMAX univarié) sur des folds.
    Stratégie : sélection du meilleur candidat (AIC) par fold et par station,
    puis prévision sur plusieurs pas.

    Retourne :
      - preds (format long)
      - metrics_global (par horizon)
      - metrics_station (par station et horizon)
      - fit_logs_df (modèle retenu, AIC et éventuels échecs)
    """
    if candidates is None:

        candidates = [
            ("arima_100", (1, 0, 0), (0, 0, 0, 0)),
            ("arima_011", (0, 1, 1), (0, 0, 0, 0)),
            ("arima_111", (1, 1, 1), (0, 0, 0, 0)),
            ("sarima_101_101_12", (1, 0, 1), (1, 0, 1, 12)),
            ("sarima_011_101_12", (0, 1, 1), (1, 0, 1, 12)),
        ]

    rows: list[dict] = []
    fit_logs: list[dict] = []
    max_h = int(max(horizons.values()))

    for _, r in folds.iterrows():
        fold_id = int(r["fold_id"])
        train_start = pd.Timestamp(r["train_start"])
        train_end = pd.Timestamp(r["train_end"])
        eval_end = pd.Timestamp(r["eval_end"])

        for sid, y in series_by_station.items():
            y = y.sort_index()

            train_start_eff = max(train_start, train_end - pd.Timedelta(days=int(max_train_days)))
            y_train = y[(y.index >= train_start_eff) & (y.index <= train_end)].copy()
            y_train = y_train.asfreq(expected)

            if train_end not in y_train.index:
                continue
            if y_train.isna().any():
                continue
            if len(y_train) < int(min_train_points):
                continue

            best_res = None
            best_name = None
            best_aic = np.inf
            fail_count = 0

            for name, order, seas in candidates:
                try:
                    res = _fit_sarimax_try(
                        y_train,
                        order=order,
                        seasonal_order=seas,
                        maxiter=maxiter,
                    )
                    aic = float(res.aic)
                    if aic < best_aic:
                        best_aic = aic
                        best_res = res
                        best_name = name
                except Exception:
                    fail_count += 1
                    continue

            if best_res is None:
                fit_logs.append({
                    "mode": mode_name,
                    "fold_id": fold_id,
                    "id_sonde": sid,
                    "train_end": train_end,
                    "chosen": "FAILED_ALL",
                    "n_train": int(len(y_train)),
                    "fails": int(fail_count),
                })
                continue

            try:
                fcst = best_res.forecast(steps=max_h)
            except Exception as e:
                fit_logs.append({
                    "mode": mode_name,
                    "fold_id": fold_id,
                    "id_sonde": sid,
                    "train_end": train_end,
                    "chosen": f"FAILED_FORECAST:{best_name}:{e}",
                    "n_train": int(len(y_train)),
                    "fails": int(fail_count),
                    "aic": float(best_aic),
                })
                continue

            fit_logs.append({
                "mode": mode_name,
                "fold_id": fold_id,
                "id_sonde": sid,
                "train_end": train_end,
                "chosen": best_name,
                "aic": float(best_aic),
                "n_train": int(len(y_train)),
                "fails": int(fail_count),
            })

            for h_name, h_steps in horizons.items():
                h_steps = int(h_steps)
                target_ts = train_end + h_steps * expected

                if target_ts > eval_end:
                    continue
                if target_ts not in y.index:
                    continue

                y_true = float(y.loc[target_ts])
                y_pred = float(fcst.iloc[h_steps - 1])

                rows.append({
                    "mode": mode_name,
                    "fold_id": fold_id,
                    "id_sonde": sid,
                    "train_end": train_end,
                    "target_ts": target_ts,
                    "horizon": h_name,
                    "model": "sarima",
                    "y_true": y_true,
                    "y_pred": y_pred,
                })

    preds = pd.DataFrame(rows)
    fit_logs_df = pd.DataFrame(fit_logs)

    if preds.empty:
        if not fit_logs_df.empty:
            print("Top chosen:")
            print(fit_logs_df["chosen"].value_counts().head(10))
        raise ValueError(f"Aucune prédiction SARIMA générée pour mode={mode_name}")

    metrics_global = (
        preds.groupby(["mode", "model", "horizon"], sort=True)[["y_true", "y_pred"]]
        .apply(lambda g: pd.Series(eval_metrics(g["y_true"], g["y_pred"])))
        .reset_index()
        .sort_values(["mode", "horizon"])
        .reset_index(drop=True)
    )

    metrics_station = (
        preds.groupby(["mode", "id_sonde", "model", "horizon"], sort=True)[["y_true", "y_pred"]]
        .apply(lambda g: pd.Series(eval_metrics(g["y_true"], g["y_pred"])))
        .reset_index()
        .sort_values(["mode", "id_sonde", "horizon"])
        .reset_index(drop=True)
    )

    counts = (
        preds.groupby(["mode", "model", "horizon"], sort=True)
        .size()
        .reset_index(name="n_preds")
    )
    metrics_global = metrics_global.merge(counts, on=["mode", "model", "horizon"], how="left")

    return preds, metrics_global, metrics_station, fit_logs_df


def _build_station_frames(df_model, *, id_col, ts_col, cols, expected):
    """
    Prépare un dictionnaire {station_id: df_station indexé par ts}
    pour effectuer les découpes plus rapidement.
    """
    frames = {}
    for sid, g in df_model[[id_col, ts_col] + cols].groupby(id_col, sort=False):
        d = g.sort_values(ts_col).set_index(ts_col)
        d = d.asfreq(expected)
        frames[int(sid)] = d
    return frames


def eval_sarimax_exog_on_folds(
    folds,
    df_model,
    *,
    id_col='id_sonde',
    ts_col='ts',
    target_col='temp_water_c',
    exog_cols,
    horizons,
    expected=pd.Timedelta(hours=2),
    mode_name='expanding',
    max_train_days=365,
    maxiter=60,
    order=(1, 0, 1),
    seasonal_order=(1, 0, 1, 12),
    min_train_points=200,
):
    """
    SARIMAX univarié avec variables exogènes futures lues depuis df_model.
    Évalue 1 origine par fold/station, prédit aux horizons demandés.

    Retour:
      - preds
      - metrics_global
      - metrics_station
      - logs
    """
    cols_needed = [target_col] + exog_cols
    station_frames = _build_station_frames(
        df_model,
        id_col=id_col,
        ts_col=ts_col,
        cols=cols_needed,
        expected=expected,
    )

    rows: list[dict] = []
    logs: list[dict] = []

    max_h = int(max(horizons.values()))

    for _, r in folds.iterrows():
        fold_id = int(r["fold_id"])
        train_start = pd.Timestamp(r["train_start"])
        train_end = pd.Timestamp(r["train_end"])
        eval_end = pd.Timestamp(r["eval_end"])

        for sid, d in station_frames.items():

            train_start_eff = max(train_start, train_end - pd.Timedelta(days=int(max_train_days)))
            dtr = d.loc[train_start_eff:train_end].copy()

            if dtr.empty:
                continue
            if train_end not in dtr.index:
                continue
            if len(dtr) < int(min_train_points):
                continue
            if dtr[target_col].isna().any():
                continue
            if dtr[exog_cols].isna().any().any():
                continue

            y_train = dtr[target_col]
            X_train = dtr[exog_cols]


            future_index = pd.date_range(
                train_end + expected,
                train_end + max_h * expected,
                freq=expected,
            )
            X_fut = d[exog_cols].reindex(future_index)

            if X_fut.isna().any().any():

                logs.append({
                    "mode": mode_name, "fold_id": fold_id, "id_sonde": sid,
                    "train_end": train_end, "status": "SKIP_MISSING_EXOG_FUT",
                    "n_train": int(len(y_train)),
                })
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)

                    mod = SARIMAX(
                        y_train,
                        exog=X_train,
                        order=order,
                        seasonal_order=seasonal_order,
                        trend="c",
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    res = mod.fit(disp=False, maxiter=int(maxiter))

                fcst = res.forecast(steps=max_h, exog=X_fut)

                logs.append({
                    "mode": mode_name, "fold_id": fold_id, "id_sonde": sid,
                    "train_end": train_end, "status": "OK",
                    "order": str(order), "seasonal_order": str(seasonal_order),
                    "n_train": int(len(y_train)),
                })
            except Exception as e:
                logs.append({
                    "mode": mode_name, "fold_id": fold_id, "id_sonde": sid,
                    "train_end": train_end, "status": f"FAILED:{str(e)[:120]}",
                    "order": str(order), "seasonal_order": str(seasonal_order),
                    "n_train": int(len(y_train)),
                })
                continue

            for h_name, h_steps in horizons.items():
                h_steps = int(h_steps)
                target_ts = train_end + h_steps * expected
                if target_ts > eval_end:
                    continue
                if target_ts not in d.index:
                    continue
                y_true = d.at[target_ts, target_col]
                if pd.isna(y_true):
                    continue

                y_pred = float(fcst.iloc[h_steps - 1])

                rows.append({
                    "mode": mode_name,
                    "fold_id": fold_id,
                    "id_sonde": sid,
                    "train_end": train_end,
                    "target_ts": target_ts,
                    "horizon": h_name,
                    "model": "sarimax",
                    "y_true": float(y_true),
                    "y_pred": y_pred,
                })

    preds = pd.DataFrame(rows)
    logs_df = pd.DataFrame(logs)

    if preds.empty:
        if not logs_df.empty:
            print("SARIMAX logs status top:")
            print(logs_df["status"].value_counts().head(10))
        raise ValueError("Aucune prédiction SARIMAX générée.")

    metrics_global = (
        preds.groupby(["mode", "model", "horizon"], sort=True)[["y_true", "y_pred"]]
        .apply(lambda g: pd.Series(eval_metrics(g["y_true"], g["y_pred"])))
        .reset_index()
        .sort_values(["mode", "horizon"])
        .reset_index(drop=True)
    )
    counts = preds.groupby(["mode", "model", "horizon"], sort=True).size().reset_index(name="n_preds")
    metrics_global = metrics_global.merge(counts, on=["mode", "model", "horizon"], how="left")

    metrics_station = (
        preds.groupby(["mode", "id_sonde", "model", "horizon"], sort=True)[["y_true", "y_pred"]]
        .apply(lambda g: pd.Series(eval_metrics(g["y_true"], g["y_pred"])))
        .reset_index()
        .sort_values(["mode", "id_sonde", "horizon"])
        .reset_index(drop=True)
    )

    return preds, metrics_global, metrics_station, logs_df


def eval_baselines_on_folds(folds, series_by_station, *, horizons, expected, mode_name='expanding'):
    """
    Baselines à origine de prévision fixe, comparables à ETS et SARIMA :
    - naive_last : y_pred = y(train_end)
    - snaive_d   : y_pred = y(target_ts - 24h)  (12 pas de 2h)
    - snaive_w   : y_pred = y(target_ts - 7j)   (84 pas de 2h)
    """
    seasonal_steps = {"snaive_d": 12, "snaive_w": 84}

    rows = []
    for _, r in folds.iterrows():
        fold_id = int(r["fold_id"])
        train_end = pd.Timestamp(r["train_end"])
        eval_end = pd.Timestamp(r["eval_end"])

        for sid, y in series_by_station.items():
            y = y.sort_index()

            if train_end not in y.index:
                continue
            y_last = y.loc[train_end]
            if pd.isna(y_last):
                continue

            for h_name, h_steps in horizons.items():
                h_steps = int(h_steps)
                target_ts = train_end + h_steps * expected

                if target_ts > eval_end:
                    continue
                if target_ts not in y.index:
                    continue
                y_true = y.loc[target_ts]
                if pd.isna(y_true):
                    continue


                rows.append({
                    "mode": mode_name, "fold_id": fold_id, "id_sonde": int(sid),
                    "train_end": train_end, "target_ts": target_ts,
                    "horizon": h_name, "model": "naive_last",
                    "y_true": float(y_true), "y_pred": float(y_last),
                })


                for mname, s in seasonal_steps.items():
                    ref_ts = target_ts - int(s) * expected
                    if ref_ts not in y.index:
                        continue
                    y_ref = y.loc[ref_ts]
                    if pd.isna(y_ref):
                        continue
                    rows.append({
                        "mode": mode_name, "fold_id": fold_id, "id_sonde": int(sid),
                        "train_end": train_end, "target_ts": target_ts,
                        "horizon": h_name, "model": mname,
                        "y_true": float(y_true), "y_pred": float(y_ref),
                    })

    preds = pd.DataFrame(rows)
    if preds.empty:
        raise ValueError("Aucune prédiction baseline générée (folds).")

    def _smape(y_true, y_pred, eps=1e-8):
        denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
        return float(200.0 * np.mean(np.abs(y_true - y_pred) / denom))

    def _eval(g):
        yt = g["y_true"].values
        yp = g["y_pred"].values
        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        smape = _smape(yt, yp)
        return pd.Series({"mae": mae, "rmse": rmse, "smape": smape})

    metrics_global = (
        preds.groupby(["mode", "model", "horizon"], sort=True)
        .apply(_eval)
        .reset_index()
    )
    counts = preds.groupby(["mode", "model", "horizon"]).size().reset_index(name="n_preds")
    metrics_global = metrics_global.merge(counts, on=["mode", "model", "horizon"], how="left")

    return preds, metrics_global


def eval_ridge_on_folds(
    folds,
    df_model,
    *,
    id_col='id_sonde',
    ts_col='ts',
    target_col='temp_water_c',
    horizons,
    expected=pd.Timedelta(hours=2),
    mode_name='expanding',
    max_train_days=365,
    alpha=1.0,
    seed=42,
    min_train_rows=200,
):
    """
    Ridge global (encodage indicateur de la station) évalué avec une origine
    de prévision par fold.
    Pour chaque fold :
      - ajuste le modèle sur la fenêtre d'entraînement (<= train_end)
        avec les labels y(t+h) disponibles dans train
      - prédit à t=train_end la valeur y(train_end + h)
    Retourne :
      - preds (format long)
      - metrics_global
      - metrics_station
      - logs_df
    """
    df = df_model.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values([id_col, ts_col]).reset_index(drop=True)

    station_ids = sorted(df[id_col].dropna().unique().tolist())

    df_future = df[[id_col, ts_col, target_col]].rename(
        columns={ts_col: "target_ts", target_col: "y_future"}
    )

    non_feat = {id_col, ts_col, target_col, "split"}
    feat_cols = [c for c in df.columns if c not in non_feat]

    non_numeric = df[feat_cols].select_dtypes(exclude=["number", "bool"]).columns.tolist()
    feat_cols_num = [c for c in feat_cols if c not in non_numeric]

    rows: list[dict] = []
    logs: list[dict] = []

    for _, r in folds.iterrows():
        fold_id = int(r["fold_id"])
        train_start = pd.Timestamp(r["train_start"])
        train_end = pd.Timestamp(r["train_end"])
        eval_end = pd.Timestamp(r["eval_end"])

        train_start_eff = max(train_start, train_end - pd.Timedelta(days=int(max_train_days)))

        df_tr_win = df[(df[ts_col] >= train_start_eff) & (df[ts_col] <= train_end)].copy()
        if df_tr_win.empty:
            continue

        for h_name, h_steps in horizons.items():
            h_steps = int(h_steps)
            delta = h_steps * expected

            tr = df_tr_win[[id_col, ts_col] + feat_cols_num].copy()
            tr["target_ts"] = tr[ts_col] + delta
            tr = tr[tr["target_ts"] <= train_end]

            tr = tr.merge(df_future, on=[id_col, "target_ts"], how="left")
            tr = tr[tr["y_future"].notna()].copy()
            tr = tr[tr[feat_cols_num].notna().all(axis=1)].copy()

            if len(tr) < int(min_train_rows):
                logs.append({
                    "mode": mode_name, "fold_id": fold_id, "horizon": h_name,
                    "status": "SKIP_TOO_FEW_TRAIN", "n_train": int(len(tr))
                })
                continue

            tr[id_col] = pd.Categorical(tr[id_col], categories=station_ids)
            Xtr = pd.get_dummies(tr[[id_col] + feat_cols_num], columns=[id_col], prefix="st", drop_first=False)
            ytr = tr["y_future"].astype(float).values

            pipe = Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("model", Ridge(alpha=float(alpha), random_state=int(seed))),
            ])
            pipe.fit(Xtr, ytr)

            df_eval = df[df[ts_col] == train_end][[id_col, ts_col] + feat_cols_num].copy()
            if df_eval.empty:
                logs.append({
                    "mode": mode_name, "fold_id": fold_id, "horizon": h_name,
                    "status": "SKIP_NO_EVAL_ROWS", "n_train": int(len(tr))
                })
                continue

            df_eval["target_ts"] = df_eval[ts_col] + delta
            df_eval = df_eval[df_eval["target_ts"] <= eval_end].copy()
            df_eval = df_eval.merge(df_future, on=[id_col, "target_ts"], how="left")
            df_eval = df_eval[df_eval["y_future"].notna()].copy()
            df_eval = df_eval[df_eval[feat_cols_num].notna().all(axis=1)].copy()

            if df_eval.empty:
                logs.append({
                    "mode": mode_name, "fold_id": fold_id, "horizon": h_name,
                    "status": "SKIP_EMPTY_EVAL", "n_train": int(len(tr))
                })
                continue

            df_eval[id_col] = pd.Categorical(df_eval[id_col], categories=station_ids)
            Xev = pd.get_dummies(df_eval[[id_col] + feat_cols_num], columns=[id_col], prefix="st", drop_first=False)

            Xev = Xev.reindex(columns=Xtr.columns, fill_value=0.0)

            y_pred = pipe.predict(Xev).astype(float)
            y_true = df_eval["y_future"].astype(float).values

            logs.append({
                "mode": mode_name, "fold_id": fold_id, "horizon": h_name,
                "status": "OK",
                "n_train": int(len(tr)),
                "n_eval": int(len(y_true)),
                "n_features": int(Xtr.shape[1]),
                "non_numeric_cols_removed": int(len(non_numeric)),
            })

            for i in range(len(df_eval)):
                rows.append({
                    "mode": mode_name,
                    "fold_id": fold_id,
                    "id_sonde": int(df_eval.iloc[i][id_col]),
                    "train_end": train_end,
                    "target_ts": pd.Timestamp(df_eval.iloc[i]["target_ts"]),
                    "horizon": h_name,
                    "model": "ridge",
                    "y_true": float(y_true[i]),
                    "y_pred": float(y_pred[i]),
                })

    preds = pd.DataFrame(rows)
    logs_df = pd.DataFrame(logs)

    if preds.empty:
        raise ValueError("Aucune prédiction Ridge générée sur folds (vérifie NA features / min_train_rows).")

    metrics_global = (
        preds.groupby(["mode", "model", "horizon"], sort=True)[["y_true", "y_pred"]]
        .apply(lambda g: pd.Series(eval_metrics(g["y_true"], g["y_pred"])))
        .reset_index()
        .sort_values(["mode", "horizon"])
        .reset_index(drop=True)
    )

    metrics_station = (
        preds.groupby(["mode", "id_sonde", "model", "horizon"], sort=True)[["y_true", "y_pred"]]
        .apply(lambda g: pd.Series(eval_metrics(g["y_true"], g["y_pred"])))
        .reset_index()
        .sort_values(["mode", "id_sonde", "horizon"])
        .reset_index(drop=True)
    )

    counts = preds.groupby(["mode", "model", "horizon"], sort=True).size().reset_index(name="n_preds")
    metrics_global = metrics_global.merge(counts, on=["mode", "model", "horizon"], how="left")

    return preds, metrics_global, metrics_station, logs_df
