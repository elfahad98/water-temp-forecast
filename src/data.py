from pathlib import Path
import pandas as pd


def parse_mixed_timestamp(series):
    """
    Convertit des horodatages provenant de formats d'entrée mixtes.
    Gère les variantes courantes de dates sous forme de chaînes.
    """
    s = series.astype(str).str.strip()

    ts = pd.to_datetime(s, errors="coerce", dayfirst=True)

    mask = ts.isna()
    if mask.any():
        ts2 = pd.to_datetime(s[mask], errors="coerce")
        ts.loc[mask] = ts2

    return ts


def prepare_base1(raw_path, interim_path, rename_map=None, raw_ts_col='ts_raw', station_col='id_sonde'):
    """
    Pipeline du notebook 1 :
    - charge le CSV
    - renomme les colonnes
    - convertit les horodatages
    - trie chronologiquement par station
    - enregistre un parquet intermédiaire
    - retourne le dataframe et un mini résumé de qualité
    """
    raw_path = Path(raw_path)
    interim_path = Path(interim_path)
    interim_path.parent.mkdir(parents=True, exist_ok=True)

    
    df = pd.read_csv(raw_path)

    if rename_map is not None:
        df = df.rename(columns=rename_map)

    if raw_ts_col not in df.columns:
        raise ValueError(f"Colonne timestamp '{raw_ts_col}' absente après rename.")

    df["ts"] = parse_mixed_timestamp(df[raw_ts_col])

    if station_col not in df.columns:
        raise ValueError(f"Colonne station '{station_col}' absente après rename.")

    df = df.sort_values([station_col, "ts"]).reset_index(drop=True)

    df.to_parquet(interim_path, index=False)

    qc = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "n_stations": int(df[station_col].nunique()),
        "min_ts": str(df["ts"].min()),
        "max_ts": str(df["ts"].max()),
        "n_ts_nat": int(df["ts"].isna().sum()),
        "n_dup_station_ts": int(df.duplicated([station_col, "ts"]).sum()),}

    return df, qc