from __future__ import annotations
import numpy as np
import pandas as pd

def build_features_v0(
    df,
    *,
    target='temp_water_c',
    station_col='id_sonde',
    ts_col='ts',
    lag_steps=(1, 12, 84),
    roll_windows=(12, 84),
):
    """
    Jeu de variables v0 :
    - variables calendaires sin/cos (heure, jour de l'année)
    - retards de la cible
    - moyenne et écart-type glissants sur la série décalée
    """
    out = df.copy()


    hour = out[ts_col].dt.hour + out[ts_col].dt.minute / 60.0
    out["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    out["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)

    doy = out[ts_col].dt.dayofyear.astype(float)
    out["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    out["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)


    for k in lag_steps:
        out[f"lag_{k}"] = out.groupby(station_col)[target].shift(k)


    shifted = out.groupby(station_col)[target].shift(1)
    for w in roll_windows:
        out[f"roll_mean_{w}"] = (
            shifted.groupby(out[station_col])
                   .rolling(window=w, min_periods=w)
                   .mean()
                   .reset_index(level=0, drop=True)
        )
        out[f"roll_std_{w}"] = (
            shifted.groupby(out[station_col])
                   .rolling(window=w, min_periods=w)
                   .std()
                   .reset_index(level=0, drop=True))

    return out
