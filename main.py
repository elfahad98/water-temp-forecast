from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.continuous_forecast import ( 
    eval_metrics_simple,
    predict_continuous_baseline,
    predict_continuous_ridge,
    predict_continuous_sarimax,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "ce script rejoue la logique "
            "d'évaluation/prédiction sur une fenêtre temporelle donnée "
            "à partir d'un dataset complet et compatible."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Chemin du fichier d'entrée (.csv ou .parquet).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Chemin du CSV de prédictions à écrire.",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "models" / "inference_config.json"),
        help="Chemin du bundle d'inférence JSON.",
    )
    parser.add_argument(
        "--horizon",
        required=True,
        choices=["h2", "d1"],
        help="Horizon de prévision à lancer.",
    )
    parser.add_argument(
        "--model",
        default="auto",
        choices=["auto", "ridge", "baseline", "sarimax"],
        help="Choix du modèle. 'auto' suit la sélection finale sauvegardée.",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Début de la fenêtre de prédiction (datetime parseable par pandas).",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="Fin de la fenêtre de prédiction (datetime parseable par pandas).",
    )
    parser.add_argument(
        "--station-id",
        dest="station_ids",
        action="append",
        type=int,
        help="Identifiant de station à traiter. Répéter l'option si besoin.",
    )
    parser.add_argument(
        "--metrics-output",
        help="Chemin optionnel du CSV de métriques globales et par station.",
    )
    parser.add_argument(
        "--stride-steps",
        type=int,
        default=1,
        help="Sous-échantillonnage des origines de prévision.",
    )
    parser.add_argument(
        "--max-train-days",
        type=int,
        default=365,
        help="Taille maximale de l'historique d'entraînement en jours.",
    )
    parser.add_argument(
        "--baseline-rule",
        default=None,
        choices=["naive", "seasonal_daily", "seasonal_weekly"],
        help="Règle baseline à utiliser pour d1 si model=baseline.",
    )
    parser.add_argument(
        "--force-2h-freq",
        action="store_true",
        help="Force une fréquence 2H stricte et échoue si des trous apparaissent.",
    )
    return parser.parse_args()


def load_bundle(config_path):
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Bundle introuvable: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_table(input_path):
    path = Path(input_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Fichier d'entrée introuvable: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError("Format non supporté. Utilise .csv ou .parquet.")
    return df, path


def resolve_model(bundle, horizon, requested_model):
    if requested_model != "auto":
        return requested_model

    if horizon == "h2":
        return bundle["h2"]["method"].replace("predict_continuous_", "")

    if horizon == "d1":
        default_method = bundle["d1"]["default_method"].replace("predict_continuous_", "")
        return "baseline" if default_method == "baseline" else default_method

    raise ValueError(f"Horizon non supporté: {horizon}")


def validate_model_choice(horizon, model):
    allowed = {
        "h2": {"ridge"},
        "d1": {"baseline", "sarimax"},
    }
    if model not in allowed[horizon]:
        choices = ", ".join(sorted(allowed[horizon]))
        raise ValueError(f"Pour l'horizon {horizon}, seuls ces modèles sont gérés ici : {choices}")


def validate_columns(df, bundle, model):
    required = {
        bundle["id_col"],
        bundle["ts_col"],
        bundle["target_col"],
    }

    if model == "ridge":
        required.update(bundle["h2"].get("exog_cols", []))
    elif model == "sarimax":
        required.update(bundle["d1"].get("exog_cols", []))

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans l'entrée: {missing}")


def normalize_table(df, bundle):
    ts_col = bundle["ts_col"]
    id_col = bundle["id_col"]

    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")
    if out[ts_col].isna().any():
        raise ValueError(f"Des dates invalides ont été détectées dans la colonne {ts_col}.")

    out = out.sort_values([id_col, ts_col]).reset_index(drop=True)
    return out


def resolve_station_ids(df, bundle, station_ids):
    id_col = bundle["id_col"]
    available = sorted(df[id_col].dropna().astype(int).unique().tolist())

    if not station_ids:
        return available

    missing = sorted(set(station_ids) - set(available))
    if missing:
        raise ValueError(f"Stations absentes du fichier d'entrée: {missing}")

    return station_ids


def run_prediction(df, bundle, args, station_id, resolved_model):
    common = {
        "df": df,
        "station_id": station_id,
        "start": args.start,
        "end": args.end,
        "ts_col": bundle["ts_col"],
        "id_col": bundle["id_col"],
        "y_col": bundle["target_col"],
    }

    if resolved_model == "ridge":
        pred = predict_continuous_ridge(
            **common,
            horizon="h2",
            exog_cols=bundle["h2"].get("exog_cols", []),
            max_train_days=args.max_train_days,
            alpha=float(bundle["h2"]["alpha"]),
            standardize=True,
        )
        label = "ridge"
        horizon = "h2"
    elif resolved_model == "baseline":
        pred = predict_continuous_baseline(
            **common,
            horizon="d1",
            rule=args.baseline_rule or bundle["d1"]["baseline_rule"],
            stride_steps=args.stride_steps,
        )
        label = f"baseline_{args.baseline_rule or bundle['d1']['baseline_rule']}"
        horizon = "d1"
    elif resolved_model == "sarimax":
        pred = predict_continuous_sarimax(
            **common,
            horizon="d1",
            order=tuple(bundle["d1"]["sarimax_order"]),
            seasonal_order=tuple(bundle["d1"]["sarimax_seasonal_order"]),
            exog_cols=bundle["d1"].get("exog_cols", []),
            max_train_days=args.max_train_days,
            stride_steps=args.stride_steps,
            force_2h_freq=args.force_2h_freq,
        )
        label = "sarimax"
        horizon = "d1"
    else:
        raise ValueError(f"Modèle non supporté: {resolved_model}")

    if pred is None or pred.empty:
        return pd.DataFrame()

    pred = pred.copy()
    pred[bundle["id_col"]] = station_id
    pred["model"] = label
    pred["horizon"] = horizon
    return pred


def build_metrics(pred_df, bundle):
    id_col = bundle["id_col"]
    rows = []

    global_metrics = eval_metrics_simple(pred_df["y_true"], pred_df["y_pred"])
    global_row = {"scope": "global"}
    global_row.update(global_metrics)
    rows.append(global_row)

    for station_id, group in pred_df.groupby(id_col):
        station_metrics = eval_metrics_simple(group["y_true"], group["y_pred"])
        station_row = {
            "scope": "station",
            id_col: station_id,
        }
        station_row.update(station_metrics)
        rows.append(station_row)

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    bundle = load_bundle(args.config)
    df, input_path = load_table(args.input)

    resolved_model = resolve_model(bundle, args.horizon, args.model)
    validate_model_choice(args.horizon, resolved_model)
    validate_columns(df, bundle, resolved_model)
    df = normalize_table(df, bundle)
    stations = resolve_station_ids(df, bundle, args.station_ids)

    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)
    if end < start:
        raise ValueError("--end doit être postérieur ou égal à --start.")

    preds = []
    for station_id in stations:
        pred = run_prediction(df, bundle, args, station_id, resolved_model)
        if not pred.empty:
            preds.append(pred)

    if not preds:
        raise RuntimeError(
            "Aucune prédiction produite. Vérifie la fenêtre temporelle, "
            "la présence d'historique suffisant et les colonnes d'entrée."
        )

    pred_df = pd.concat(preds, ignore_index=True)
    pred_df = pred_df.sort_values([bundle["id_col"], "target_ts"]).reset_index(drop=True)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, index=False)

    print(f"Input loaded : {input_path}")
    print(f"Model used   : {resolved_model}")
    print(f"Horizon      : {args.horizon}")
    print(f"Stations     : {stations}")
    print(f"Saved preds  : {output_path}")

    metrics_df = build_metrics(pred_df, bundle)
    print("\nMetrics:")
    print(metrics_df.to_string(index=False))

    if args.metrics_output:
        metrics_path = Path(args.metrics_output).resolve()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()