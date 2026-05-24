from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import pandas as pd

from typing import Iterable, Optional, Dict, Tuple, List

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings

import warnings
warnings.filterwarnings(
    "ignore",
    message="No frequency information was provided, so inferred frequency",)


HORIZON_TO_STEPS = {"h2": 1, "d1": 12, "w1": 84}


def _check_required_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def _as_2d_float(x):
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

def _as_1d_float(x):
    return np.asarray(x, dtype=float).reshape(-1,)

def _prepare_station_df(df, station_id, ts_col, id_col, cols_needed, force_2h_freq=False):
    _check_required_cols(df, [ts_col, id_col] + cols_needed)

    d = df[df[id_col] == station_id].copy()
    d[ts_col] = pd.to_datetime(d[ts_col])
    d = d.sort_values(ts_col).set_index(ts_col)


    d.index = pd.DatetimeIndex(d.index)


    diffs = d.index.to_series().diff().dropna()
    if not diffs.empty and (diffs == pd.Timedelta(hours=2)).all():
        d.index.freq = "2H"


    if force_2h_freq:
        before_na = d[cols_needed].isna().sum().sum()
        d = d.asfreq("2H")
        after_na = d[cols_needed].isna().sum().sum()
        if after_na > before_na:
            raise ValueError(
                "force_2h_freq=True introduced NA values (gaps exist). "
                "Set force_2h_freq=False or fix gaps first."
            )

    return d


def predict_continuous_sarimax(
    df,
    station_id,
    start,
    end,
    ts_col,
    id_col,
    y_col,
    horizon='h2',
    order=(1, 0, 1),
    seasonal_order=(1, 0, 1, 12),
    exog_cols=None,
    max_train_days=365,
    stride_steps=1,
    force_2h_freq=False,
):
    """
    Prédiction continue en avance pas à pas.
    - Ajuste une fois sur la fenêtre d'entraînement (max_train_days avant le départ)
    - Met à jour l'état du filtre avec append(refit=False) à chaque origine
    - Produit une prévision à chaque origine pour l'horizon demandé

    Retourne un DataFrame avec les colonnes : ['target_ts', 'y_true', 'y_pred']
    """
    if horizon not in HORIZON_TO_STEPS:
        raise ValueError(f"Unknown horizon={horizon}. Use one of {list(HORIZON_TO_STEPS)}")

    steps = HORIZON_TO_STEPS[horizon]
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    cols_needed = [y_col] + (exog_cols if exog_cols else [])
    d = _prepare_station_df(df, station_id, ts_col, id_col, cols_needed, force_2h_freq=force_2h_freq)

    train_start = start - pd.Timedelta(days=max_train_days)
    train_end = start - pd.Timedelta(hours=2)
    train = d.loc[train_start:train_end]

    y_train = train[y_col].astype(float)
    X_train = train[exog_cols].astype(float) if exog_cols else None

    mod = SARIMAX(
        y_train,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = mod.fit(disp=False)

    idx_all = d.loc[start:end].index
    idx = idx_all[::stride_steps] if stride_steps > 1 else idx_all

    rows = []

    for t in idx:
        t_target = t + pd.Timedelta(hours=2 * steps)


        if t_target not in d.index:
            if exog_cols:
                endog_new = _as_1d_float([d.loc[t, y_col]])
                exog_new = _as_2d_float(d.loc[[t], exog_cols].values)
                if np.any(~np.isfinite(exog_new)) or np.any(~np.isfinite(endog_new)):
                    continue
                res = res.append(endog_new, exog=exog_new, refit=False)
            else:
                endog_new = _as_1d_float([d.loc[t, y_col]])
                if np.any(~np.isfinite(endog_new)):
                    continue
                res = res.append(endog_new, refit=False)
            continue


        if exog_cols:
            t_last_exog = t + pd.Timedelta(hours=2 * (steps - 1))
            X_future_df = d.loc[t:t_last_exog, exog_cols]
            if len(X_future_df) != steps:

                endog_new = _as_1d_float([d.loc[t, y_col]])
                exog_new = _as_2d_float(d.loc[[t], exog_cols].values)
                if np.any(~np.isfinite(exog_new)) or np.any(~np.isfinite(endog_new)):
                    continue
                res = res.append(endog_new, exog=exog_new, refit=False)
                continue

            X_future = _as_2d_float(X_future_df.values)
            if np.any(~np.isfinite(X_future)):
                endog_new = _as_1d_float([d.loc[t, y_col]])
                exog_new = _as_2d_float(d.loc[[t], exog_cols].values)
                if np.any(~np.isfinite(exog_new)) or np.any(~np.isfinite(endog_new)):
                    continue
                res = res.append(endog_new, exog=exog_new, refit=False)
                continue

            fc = res.get_forecast(steps=steps, exog=X_future)
        else:
            fc = res.get_forecast(steps=steps)

        yhat = float(fc.predicted_mean.iloc[-1])
        ytrue = float(d.loc[t_target, y_col])
        rows.append((t_target, ytrue, yhat))


        if exog_cols:
            endog_new = _as_1d_float([d.loc[t, y_col]])
            exog_new = _as_2d_float(d.loc[[t], exog_cols].values)
            if np.any(~np.isfinite(exog_new)) or np.any(~np.isfinite(endog_new)):
                continue
            res = res.append(endog_new, exog=exog_new, refit=False)
        else:
            endog_new = _as_1d_float([d.loc[t, y_col]])
            if np.any(~np.isfinite(endog_new)):
                continue
            res = res.append(endog_new, refit=False)

    return pd.DataFrame(rows, columns=["target_ts", "y_true", "y_pred"]).sort_values("target_ts")


def predict_continuous_sarima(
    df,
    station_id,
    start,
    end,
    ts_col,
    id_col,
    y_col,
    horizon='h2',
    order=(1, 0, 1),
    seasonal_order=(1, 0, 1, 12),
    max_train_days=365,
    stride_steps=1,
    force_2h_freq=False,
):
    return predict_continuous_sarimax(
        df=df,
        station_id=station_id,
        start=start,
        end=end,
        ts_col=ts_col,
        id_col=id_col,
        y_col=y_col,
        horizon=horizon,
        order=order,
        seasonal_order=seasonal_order,
        exog_cols=None,
        max_train_days=max_train_days,
        stride_steps=stride_steps,
        force_2h_freq=force_2h_freq,
    )


def _build_ridge_frame_station(
    df,
    station_id,
    ts_col,
    id_col,
    y_col,
    horizon,
    exog_cols=None,
    y_lags=(1, 2, 3, 6, 12, 24, 84),
    exog_lags=(0, 1, 2, 6, 12),
    add_roll=True,
):
    if horizon not in HORIZON_TO_STEPS:
        raise ValueError(f"Unknown horizon={horizon}. Use one of {list(HORIZON_TO_STEPS)}")
    steps = HORIZON_TO_STEPS[horizon]
    exog_cols = exog_cols or []

    cols_needed = [y_col] + exog_cols
    d = _prepare_station_df(df, station_id, ts_col, id_col, cols_needed, force_2h_freq=False)

    X = pd.DataFrame(index=d.index)
    y = d[y_col].astype(float)


    hour = X.index.hour
    doy = X.index.dayofyear
    dow = X.index.dayofweek

    X["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    X["cos_hour"] = np.cos(2 * np.pi * hour / 24)
    X["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    X["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    X["is_weekend"] = (dow >= 5).astype(int)


    for k in y_lags:
        X[f"y_lag_{k}"] = y.shift(k)


    if add_roll:
        X["y_roll_mean_12"] = y.shift(1).rolling(12, min_periods=12).mean()
        X["y_roll_std_12"] = y.shift(1).rolling(12, min_periods=12).std()
        X["y_roll_mean_84"] = y.shift(1).rolling(84, min_periods=84).mean()


    for col in exog_cols:
        s = d[col].astype(float)
        for k in exog_lags:
            X[f"{col}_lag_{k}"] = s.shift(k)


    y_target = y.shift(-steps)

    frame = X.copy()
    frame["y_target"] = y_target
    frame = frame.dropna()
    frame["target_ts"] = frame.index + pd.Timedelta(hours=2 * steps)
    return frame


def predict_continuous_ridge(
    df,
    station_id,
    start,
    end,
    ts_col,
    id_col,
    y_col,
    horizon='h2',
    exog_cols=None,
    max_train_days=365,
    alpha=1.0,
    standardize=True,
):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    frame = _build_ridge_frame_station(
        df=df,
        station_id=station_id,
        ts_col=ts_col,
        id_col=id_col,
        y_col=y_col,
        horizon=horizon,
        exog_cols=exog_cols,
    )

    train_start = start - pd.Timedelta(days=max_train_days)
    train_end = start - pd.Timedelta(hours=2)
    train = frame.loc[train_start:train_end].copy()
    test = frame.loc[start:end].copy()

    feature_cols = [c for c in frame.columns if c not in ("y_target", "target_ts")]

    Xtr = train[feature_cols].values
    ytr = train["y_target"].values
    Xte = test[feature_cols].values
    yte = test["y_target"].values

    if standardize:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

    model = Ridge(alpha=alpha, random_state=0)
    model.fit(Xtr, ytr)
    yhat = model.predict(Xte)

    return pd.DataFrame({
        "target_ts": test["target_ts"].values,
        "y_true": yte,
        "y_pred": yhat
    }).sort_values("target_ts")


def predict_continuous_ets_refit(
    df,
    station_id,
    start,
    end,
    ts_col,
    id_col,
    y_col,
    horizon='h2',
    max_train_days=365,
    seasonal_periods=12,
    trend='add',
    seasonal='add',
    damped_trend=False,
    refit_every_steps=12,
    stride_steps=1,
    force_2h_freq=False,
):
    """
    ETS ne permet pas une mise à jour incrémentale légère comme SARIMAX.
    Stratégie : réajuster périodiquement le modèle (refit_every_steps)
    sur une fenêtre glissante
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    if horizon not in HORIZON_TO_STEPS:
        raise ValueError(f"Unknown horizon={horizon}. Use one of {list(HORIZON_TO_STEPS)}")
    steps = HORIZON_TO_STEPS[horizon]

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    d = _prepare_station_df(df, station_id, ts_col, id_col, [y_col], force_2h_freq=force_2h_freq)
    y = d[y_col].astype(float)

    idx_all = d.loc[start:end].index
    idx = idx_all[::stride_steps] if stride_steps > 1 else idx_all

    rows = []
    fit_res = None
    last_fit_t = None

    for i, t in enumerate(idx):

        need_refit = (fit_res is None) or (last_fit_t is None) or ((i % refit_every_steps) == 0)

        if need_refit:
            train_start = t - pd.Timedelta(days=max_train_days)
            train_end = t - pd.Timedelta(hours=2)
            y_train = y.loc[train_start:train_end].dropna()
            if len(y_train) < (seasonal_periods * 2):
                continue

            model = ExponentialSmoothing(
                y_train,
                trend=trend,
                damped_trend=damped_trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            )
            fit_res = model.fit(optimized=True)
            last_fit_t = t

        t_target = t + pd.Timedelta(hours=2 * steps)
        if t_target not in y.index:
            continue

        fc = fit_res.forecast(steps=steps)
        yhat = float(fc.iloc[-1])
        ytrue = float(y.loc[t_target])
        rows.append((t_target, ytrue, yhat))

    return pd.DataFrame(rows, columns=["target_ts", "y_true", "y_pred"]).sort_values("target_ts")


def plot_pred_vs_true(
    pred_df,
    title,
    zoom_days=14,
    monthly_interval=1,
    zoom_day_interval=2,
    date_fmt_full='%b %Y',
    date_fmt_zoom='%d %b',
    rotate=0,
):
    """
    pred_df doit contenir les colonnes : ['target_ts', 'y_true', 'y_pred']
    """
    if pred_df is None or pred_df.empty:
        raise ValueError("pred_df is empty — no predictions to plot.")

    d = pred_df.copy()
    d["target_ts"] = pd.to_datetime(d["target_ts"])
    d = d.sort_values("target_ts")


    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(d["target_ts"], d["y_true"], label="Réel")
    ax.plot(d["target_ts"], d["y_pred"], label="Prévision")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=monthly_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt_full))
    fig.autofmt_xdate(rotation=rotate)
    plt.show()


    if zoom_days and zoom_days > 0:
        z0 = d["target_ts"].max() - pd.Timedelta(days=zoom_days)
        z = d[d["target_ts"] >= z0].copy()
        if not z.empty:
            fig, ax = plt.subplots(figsize=(13, 4))
            ax.plot(z["target_ts"], z["y_true"], label="Réel")
            ax.plot(z["target_ts"], z["y_pred"], label="Prévision")
            ax.set_title(title + f" (zoom {zoom_days}j)")
            ax.grid(alpha=0.3)
            ax.legend()

            ax.xaxis.set_major_locator(mdates.DayLocator(interval=zoom_day_interval))
            ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt_zoom))
            fig.autofmt_xdate(rotation=rotate)
            plt.show()


def predict_continuous_baseline(
    df,
    station_id,
    start,
    end,
    ts_col,
    id_col,
    y_col,
    horizon='d1',
    rule='seasonal_daily',
    stride_steps=1,
):
    steps = HORIZON_TO_STEPS[horizon]

    d = df[df[id_col] == station_id].copy()
    d[ts_col] = pd.to_datetime(d[ts_col])
    d = d.sort_values(ts_col).set_index(ts_col)

    idx = d.loc[pd.to_datetime(start):pd.to_datetime(end)].index
    if stride_steps > 1:
        idx = idx[::stride_steps]

    rows = []

    for t in idx:
        target_ts = t + pd.Timedelta(hours=2 * steps)
        if target_ts not in d.index:
            continue

        if rule == "naive":
            ref_ts = t
        elif rule == "seasonal_daily":
            ref_ts = target_ts - pd.Timedelta(days=1)
        elif rule == "seasonal_weekly":
            ref_ts = target_ts - pd.Timedelta(days=7)
        else:
            raise ValueError(f"Unknown baseline rule: {rule}")

        if ref_ts not in d.index:
            continue

        rows.append({
            id_col: station_id,
            "target_ts": target_ts,
            "y_true": float(d.loc[target_ts, y_col]),
            "y_pred": float(d.loc[ref_ts, y_col]),
            "model": f"baseline_{rule}",
            "horizon": horizon,
        })

    return pd.DataFrame(rows)


def eval_metrics_simple(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = np.mean(np.where(denom == 0, 0.0, np.abs(y_true - y_pred) / denom)) * 100.0

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "smape": float(smape),
        "r2": float(r2) if pd.notna(r2) else np.nan,
        "n_preds": int(len(y_true)),
    }


def summarize_preds(pred_df, id_col='id_sonde'):
    g = eval_metrics_simple(pred_df["y_true"], pred_df["y_pred"])
    out_global = pd.DataFrame([g])

    rows = []
    for st, d in pred_df.groupby(id_col):
        m = eval_metrics_simple(d["y_true"], d["y_pred"])
        m[id_col] = st
        rows.append(m)

    out_station = pd.DataFrame(rows).sort_values(id_col).reset_index(drop=True)
    return out_global, out_station


def evaluate_final_test_models(
    df_model,
    *,
    stations,
    test_start,
    test_end,
    ts_col,
    id_col,
    y_col,
    exog_cols,
    ridge_alpha=10.0,
    baseline_rule='seasonal_daily',
    sarimax_order=(2, 0, 1),
    sarimax_seasonal_order=(1, 0, 1, 12),
    stride_test=1,
    force_2h_freq=False,
):
    preds_test_h2_ridge = []
    preds_test_d1_base = []
    preds_test_d1_smx = []

    for st in stations:

        p_ridge = predict_continuous_ridge(
            df=df_model,
            station_id=st,
            start=test_start,
            end=test_end,
            ts_col=ts_col,
            id_col=id_col,
            y_col=y_col,
            horizon="h2",
            exog_cols=exog_cols,
            max_train_days=365,
            alpha=ridge_alpha,
            standardize=True,
        )
        if p_ridge is not None and not p_ridge.empty:
            p_ridge[id_col] = st
            p_ridge["model"] = "ridge"
            p_ridge["horizon"] = "h2"
            preds_test_h2_ridge.append(p_ridge)


        p_base = predict_continuous_baseline(
            df=df_model,
            station_id=st,
            start=test_start,
            end=test_end,
            ts_col=ts_col,
            id_col=id_col,
            y_col=y_col,
            horizon="d1",
            rule=baseline_rule,
            stride_steps=stride_test,
        )
        if p_base is not None and not p_base.empty:
            preds_test_d1_base.append(p_base)


        p_smx = predict_continuous_sarimax(
            df=df_model,
            station_id=st,
            start=test_start,
            end=test_end,
            ts_col=ts_col,
            id_col=id_col,
            y_col=y_col,
            horizon="d1",
            order=sarimax_order,
            seasonal_order=sarimax_seasonal_order,
            exog_cols=exog_cols,
            max_train_days=365,
            stride_steps=stride_test,
            force_2h_freq=force_2h_freq,
        )
        if p_smx is not None and not p_smx.empty:
            p_smx[id_col] = st
            p_smx["model"] = "sarimax"
            p_smx["horizon"] = "d1"
            preds_test_d1_smx.append(p_smx)

    preds_test_h2_ridge = pd.concat(preds_test_h2_ridge, ignore_index=True) if preds_test_h2_ridge else pd.DataFrame()
    preds_test_d1_base = pd.concat(preds_test_d1_base, ignore_index=True) if preds_test_d1_base else pd.DataFrame()
    preds_test_d1_smx = pd.concat(preds_test_d1_smx, ignore_index=True) if preds_test_d1_smx else pd.DataFrame()

    test_h2_ridge_global, test_h2_ridge_station = summarize_preds(preds_test_h2_ridge, id_col=id_col)
    test_d1_base_global, test_d1_base_station = summarize_preds(preds_test_d1_base, id_col=id_col)
    test_d1_smx_global, test_d1_smx_station = summarize_preds(preds_test_d1_smx, id_col=id_col)

    final_test_cmp = pd.concat([
        test_h2_ridge_global.assign(model="ridge", horizon="h2"),
        test_d1_base_global.assign(model=f"baseline_{baseline_rule}", horizon="d1"),
        test_d1_smx_global.assign(model="sarimax_tuned", horizon="d1"),
    ], ignore_index=True)

    final_test_cmp = final_test_cmp[["model", "horizon", "mae", "rmse", "smape", "r2", "n_preds"]]

    return {
        "preds_test_h2_ridge": preds_test_h2_ridge,
        "preds_test_d1_base": preds_test_d1_base,
        "preds_test_d1_smx": preds_test_d1_smx,
        "test_h2_ridge_global": test_h2_ridge_global,
        "test_h2_ridge_station": test_h2_ridge_station,
        "test_d1_base_global": test_d1_base_global,
        "test_d1_base_station": test_d1_base_station,
        "test_d1_smx_global": test_d1_smx_global,
        "test_d1_smx_station": test_d1_smx_station,
        "final_test_cmp": final_test_cmp,
    }


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


HORIZON_TO_STEPS = {"h2": 1, "d1": 12, "w1": 84}


def predict_continuous_baseline(
    df,
    station_id,
    start,
    end,
    ts_col,
    id_col,
    y_col,
    horizon='d1',
    rule='seasonal_daily',
    stride_steps=1,
):
    steps = HORIZON_TO_STEPS[horizon]

    d = df[df[id_col] == station_id].copy()
    d[ts_col] = pd.to_datetime(d[ts_col])
    d = d.sort_values(ts_col).set_index(ts_col)

    idx = d.loc[pd.to_datetime(start):pd.to_datetime(end)].index
    if stride_steps > 1:
        idx = idx[::stride_steps]

    rows = []

    for t in idx:
        target_ts = t + pd.Timedelta(hours=2 * steps)
        if target_ts not in d.index:
            continue

        if rule == "naive":
            ref_ts = t
        elif rule == "seasonal_daily":
            ref_ts = target_ts - pd.Timedelta(days=1)
        elif rule == "seasonal_weekly":
            ref_ts = target_ts - pd.Timedelta(days=7)
        else:
            raise ValueError(f"Unknown baseline rule: {rule}")

        if ref_ts not in d.index:
            continue

        rows.append({
            id_col: station_id,
            "target_ts": target_ts,
            "y_true": float(d.loc[target_ts, y_col]),
            "y_pred": float(d.loc[ref_ts, y_col]),
            "model": f"baseline_{rule}",
            "horizon": horizon,
        })

    return pd.DataFrame(rows)


def eval_metrics_simple(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = np.mean(np.where(denom == 0, 0.0, np.abs(y_true - y_pred) / denom)) * 100.0

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "smape": float(smape),
        "r2": float(r2) if pd.notna(r2) else np.nan,
        "n_preds": int(len(y_true)),
    }


def summarize_preds(pred_df, id_col='id_sonde'):
    g = eval_metrics_simple(pred_df["y_true"], pred_df["y_pred"])
    out_global = pd.DataFrame([g])

    rows = []
    for st, d in pred_df.groupby(id_col):
        m = eval_metrics_simple(d["y_true"], d["y_pred"])
        m[id_col] = st
        rows.append(m)

    out_station = pd.DataFrame(rows).sort_values(id_col).reset_index(drop=True)
    return out_global, out_station


def evaluate_final_test_models(
    df_model,
    *,
    stations,
    test_start,
    test_end,
    ts_col,
    id_col,
    y_col,
    exog_cols,
    ridge_alpha=10.0,
    baseline_rule='seasonal_daily',
    sarimax_order=(2, 0, 1),
    sarimax_seasonal_order=(1, 0, 1, 12),
    stride_test=1,
    force_2h_freq=False,
):
    preds_test_h2_ridge = []
    preds_test_d1_base = []
    preds_test_d1_smx = []

    for st in stations:
        p_ridge = predict_continuous_ridge(
            df=df_model,
            station_id=st,
            start=test_start,
            end=test_end,
            ts_col=ts_col,
            id_col=id_col,
            y_col=y_col,
            horizon="h2",
            exog_cols=exog_cols,
            max_train_days=365,
            alpha=ridge_alpha,
            standardize=True,
        )
        if p_ridge is not None and not p_ridge.empty:
            p_ridge[id_col] = st
            p_ridge["model"] = "ridge"
            p_ridge["horizon"] = "h2"
            preds_test_h2_ridge.append(p_ridge)

        p_base = predict_continuous_baseline(
            df=df_model,
            station_id=st,
            start=test_start,
            end=test_end,
            ts_col=ts_col,
            id_col=id_col,
            y_col=y_col,
            horizon="d1",
            rule=baseline_rule,
            stride_steps=stride_test,
        )
        if p_base is not None and not p_base.empty:
            preds_test_d1_base.append(p_base)

        p_smx = predict_continuous_sarimax(
            df=df_model,
            station_id=st,
            start=test_start,
            end=test_end,
            ts_col=ts_col,
            id_col=id_col,
            y_col=y_col,
            horizon="d1",
            order=sarimax_order,
            seasonal_order=sarimax_seasonal_order,
            exog_cols=exog_cols,
            max_train_days=365,
            stride_steps=stride_test,
            force_2h_freq=force_2h_freq,
        )
        if p_smx is not None and not p_smx.empty:
            p_smx[id_col] = st
            p_smx["model"] = "sarimax"
            p_smx["horizon"] = "d1"
            preds_test_d1_smx.append(p_smx)

    preds_test_h2_ridge = pd.concat(preds_test_h2_ridge, ignore_index=True) if preds_test_h2_ridge else pd.DataFrame()
    preds_test_d1_base = pd.concat(preds_test_d1_base, ignore_index=True) if preds_test_d1_base else pd.DataFrame()
    preds_test_d1_smx = pd.concat(preds_test_d1_smx, ignore_index=True) if preds_test_d1_smx else pd.DataFrame()

    test_h2_ridge_global, test_h2_ridge_station = summarize_preds(preds_test_h2_ridge, id_col=id_col)
    test_d1_base_global, test_d1_base_station = summarize_preds(preds_test_d1_base, id_col=id_col)
    test_d1_smx_global, test_d1_smx_station = summarize_preds(preds_test_d1_smx, id_col=id_col)

    final_test_cmp = pd.concat([
        test_h2_ridge_global.assign(model="ridge", horizon="h2"),
        test_d1_base_global.assign(model=f"baseline_{baseline_rule}", horizon="d1"),
        test_d1_smx_global.assign(model="sarimax_tuned", horizon="d1"),
    ], ignore_index=True)

    final_test_cmp = final_test_cmp[["model", "horizon", "mae", "rmse", "smape", "r2", "n_preds"]]

    return {
        "preds_test_h2_ridge": preds_test_h2_ridge,
        "preds_test_d1_base": preds_test_d1_base,
        "preds_test_d1_smx": preds_test_d1_smx,
        "test_h2_ridge_global": test_h2_ridge_global,
        "test_h2_ridge_station": test_h2_ridge_station,
        "test_d1_base_global": test_d1_base_global,
        "test_d1_base_station": test_d1_base_station,
        "test_d1_smx_global": test_d1_smx_global,
        "test_d1_smx_station": test_d1_smx_station,
        "final_test_cmp": final_test_cmp,
    }


def plot_test_forecast(df_pred, title, zoom_days=14, save_dir=None, save_stem=None):
    d = df_pred.copy()
    d["target_ts"] = pd.to_datetime(d["target_ts"])
    d = d.sort_values("target_ts")

    fig = plt.figure(figsize=(13, 4))
    plt.plot(d["target_ts"], d["y_true"], label="Réel")
    plt.plot(d["target_ts"], d["y_pred"], label="Prévision")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    if save_dir is not None and save_stem is not None:
        fig.savefig(save_dir / f"{save_stem}_full.png", dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    z0 = d["target_ts"].max() - pd.Timedelta(days=zoom_days)
    z = d[d["target_ts"] >= z0]

    fig = plt.figure(figsize=(13, 4))
    plt.plot(z["target_ts"], z["y_true"], label="Réel")
    plt.plot(z["target_ts"], z["y_pred"], label="Prévision")
    plt.title(f"{title} (zoom {zoom_days}j)")
    plt.grid(alpha=0.3)
    plt.legend()
    if save_dir is not None and save_stem is not None:
        fig.savefig(save_dir / f"{save_stem}_zoom{zoom_days}d.png", dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_compare_d1(
    df_base,
    df_smx,
    station_id=817,
    id_col='id_sonde',
    zoom_days=14,
    save_dir=None,
    save_stem=None,
):
    b = df_base[df_base[id_col] == station_id].copy()
    s = df_smx[df_smx[id_col] == station_id].copy()

    b["target_ts"] = pd.to_datetime(b["target_ts"])
    s["target_ts"] = pd.to_datetime(s["target_ts"])

    d = b[["target_ts", "y_true", "y_pred"]].rename(columns={"y_pred": "y_pred_base"})
    d = d.merge(
        s[["target_ts", "y_pred"]].rename(columns={"y_pred": "y_pred_smx"}),
        on="target_ts",
        how="inner"
    ).sort_values("target_ts")

    fig = plt.figure(figsize=(13, 4))
    plt.plot(d["target_ts"], d["y_true"], label="Réel")
    plt.plot(d["target_ts"], d["y_pred_base"], label="Baseline d1")
    plt.plot(d["target_ts"], d["y_pred_smx"], label="SARIMAX d1")
    plt.title(f"d1 — station {station_id} — TEST")
    plt.grid(alpha=0.3)
    plt.legend()
    if save_dir is not None and save_stem is not None:
        fig.savefig(save_dir / f"{save_stem}_full.png", dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    z0 = d["target_ts"].max() - pd.Timedelta(days=zoom_days)
    z = d[d["target_ts"] >= z0]

    fig = plt.figure(figsize=(13, 4))
    plt.plot(z["target_ts"], z["y_true"], label="Réel")
    plt.plot(z["target_ts"], z["y_pred_base"], label="Baseline d1")
    plt.plot(z["target_ts"], z["y_pred_smx"], label="SARIMAX d1")
    plt.title(f"d1 — station {station_id} — TEST (zoom {zoom_days}j)")
    plt.grid(alpha=0.3)
    plt.legend()
    if save_dir is not None and save_stem is not None:
        fig.savefig(save_dir / f"{save_stem}_zoom{zoom_days}d.png", dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_scatter_true_pred(df_pred, title, save_dir=None, save_name=None):
    d = df_pred.copy()

    fig = plt.figure(figsize=(5.5, 5.5))
    plt.scatter(d["y_true"], d["y_pred"], alpha=0.5)
    vmin = min(d["y_true"].min(), d["y_pred"].min())
    vmax = max(d["y_true"].max(), d["y_pred"].max())
    plt.plot([vmin, vmax], [vmin, vmax], linestyle="--")
    plt.xlabel("Réel")
    plt.ylabel("Prévision")
    plt.title(title)
    plt.grid(alpha=0.3)
    if save_dir is not None and save_name is not None:
        fig.savefig(save_dir / save_name, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)
