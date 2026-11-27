"""Train and use a booking cancellation model with a chronological evaluation."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = [
    "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies", "is_repeated_guest",
    "previous_cancellations", "previous_bookings_not_canceled",
]
CATEGORICAL = [
    "hotel", "arrival_date_month", "meal", "market_segment", "distribution_channel",
    "reserved_room_type", "customer_type",
]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "is_canceled"
VALIDATION_START = pd.Timestamp("2016-07-01")
TEST_START = pd.Timestamp("2017-01-01")


def features(frame):
    missing = sorted(set(FEATURES) - set(frame))
    if missing:
        raise ValueError(f"Missing input columns: {', '.join(missing)}")
    return frame[FEATURES]


def prepare_bookings(frame):
    arrival = pd.to_datetime(
        frame["arrival_date_year"].astype(str) + " " + frame["arrival_date_month"] + " " +
        frame["arrival_date_day_of_month"].astype(str),
        format="%Y %B %d",
    )
    booking = arrival - pd.to_timedelta(frame["lead_time"], unit="D")
    resolved = pd.to_datetime(frame["reservation_status_date"], format="mixed")
    valid = frame["lead_time"].ge(0) & resolved.ge(booking)
    return frame.loc[valid].assign(booking_date=booking[valid], resolved_date=resolved[valid])


def known_labels(frame, cutoff):
    return frame.loc[frame["booking_date"].lt(cutoff) & frame["resolved_date"].lt(cutoff)]


def candidate_models():
    preprocessing = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("categorical", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ), CATEGORICAL),
    ])
    models = {
        "base_rate": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(max_iter=500, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=80, max_depth=12, min_samples_leaf=12, random_state=42, n_jobs=2,
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=160, max_leaf_nodes=31, learning_rate=0.08, early_stopping=False,
            random_state=42,
        ),
    }
    return {name: make_pipeline(clone(preprocessing), model) for name, model in models.items()}


def score(model, frame):
    observed = frame[TARGET]
    probabilities = model.predict_proba(features(frame))[:, 1]
    return {
        "log_loss": log_loss(observed, probabilities),
        "brier": brier_score_loss(observed, probabilities),
        "roc_auc": roc_auc_score(observed, probabilities),
        "average_precision": average_precision_score(observed, probabilities),
    }


def train(csv_path, output):
    raw = pd.read_csv(csv_path)
    features(raw)
    bookings = prepare_bookings(raw)
    validation_training = known_labels(bookings, VALIDATION_START)
    validation = bookings.loc[
        bookings["booking_date"].ge(VALIDATION_START)
        & bookings["booking_date"].lt(TEST_START)
        & bookings["resolved_date"].lt(TEST_START)
    ]
    final_training = known_labels(bookings, TEST_START)
    test = bookings.loc[bookings["booking_date"].ge(TEST_START)]
    if any(frame[TARGET].nunique() != 2 for frame in (validation_training, validation, final_training, test)):
        raise ValueError("Each temporal split must contain both outcomes.")

    candidates = candidate_models()
    validation_scores = {}
    for name, model in candidates.items():
        print(f"Validating {name}", flush=True)
        model.fit(features(validation_training), validation_training[TARGET])
        validation_scores[name] = score(model, validation)
    selected = min(validation_scores, key=lambda name: validation_scores[name]["log_loss"])

    final_scores = {}
    for name, model in candidates.items():
        print(f"Evaluating {name} on the holdout", flush=True)
        model.fit(features(final_training), final_training[TARGET])
        final_scores[name] = score(model, test)

    metrics = {
        "dataset_rows": len(raw),
        "excluded_temporal_inconsistencies": len(raw) - len(bookings),
        "validation_start": str(VALIDATION_START.date()),
        "test_start": str(TEST_START.date()),
        "validation_train_rows": len(validation_training),
        "validation_rows": len(validation),
        "final_train_rows": len(final_training),
        "test_rows": len(test),
        "test_cancelled_fraction": float(test[TARGET].mean()),
        "selected_by_validation_log_loss": selected,
        "validation": validation_scores,
        "test": final_scores,
    }
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidates[selected], output / "model.joblib", compress=3)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def predict(model_path, csv_path):
    model = joblib.load(model_path)
    inputs = pd.read_csv(csv_path)
    probabilities = model.predict_proba(features(inputs))[:, 1]
    for row, value in enumerate(probabilities, 1):
        print(json.dumps({"row": row, "cancellation_probability": round(float(value), 6)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("csv", type=Path)
    training.add_argument("--output", type=Path, default=Path("artifacts"))
    inference = commands.add_parser("predict")
    inference.add_argument("model", type=Path)
    inference.add_argument("csv", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        train(args.csv, args.output)
    else:
        predict(args.model, args.csv)


if __name__ == "__main__":
    main()
