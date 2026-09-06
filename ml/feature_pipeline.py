# pyrefly: ignore [missing-import]
"""
Build the AgriFL preprocessing artifacts and federated client shards.

This pipeline keeps `data/raw/primary.csv` as the authoritative training source,
fits all transforms on the training split only, and generates IID plus
soil-based Non-IID shards from training records only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_FILES = ("primary.csv", "analysis.csv", "validation.csv")
TARGET_COLUMN = "CROPS"
FINAL_FEATURES = [
    "SOIL",
    "SEASON",
    "SOWN",
    "WATER_SOURCE",
    "SOIL_PH",
    "CROPDURATION",
    "TEMP",
    "WATERREQUIRED",
    "RELATIVE_HUMIDITY",
    "N",
    "P",
    "K",
]
CATEGORICAL_FEATURES = ["SOIL", "SEASON", "SOWN", "WATER_SOURCE"]
NUMERIC_FEATURES = [
    "SOIL_PH",
    "CROPDURATION",
    "TEMP",
    "WATERREQUIRED",
    "RELATIVE_HUMIDITY",
    "N",
    "P",
    "K",
]
LEAKAGE_COLUMNS = [
    "TYPE_OF_CROP",
    "HARVESTED",
    "SOIL_PH_HIGH",
    "CROPDURATION_MAX",
    "MAX_TEMP",
    "WATERREQUIRED_MAX",
    "RELATIVE_HUMIDITY_MAX",
    "N_MAX",
    "P_MAX",
    "K_MAX",
]
CLIENT_COUNTS = (5, 10, 20)


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )


def inspect_csv(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    target_candidates = [
        col for col in ("CROPS", "label", "Crop") if col in df.columns
    ]
    target = target_candidates[0] if target_candidates else None
    leakage_present = [col for col in LEAKAGE_COLUMNS if col in df.columns]
    summary = {
        "file": str(path.relative_to(PROJECT_ROOT)),
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {
            col: int(count)
            for col, count in df.isna().sum().items()
            if int(count) > 0
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "target_column": target,
        "class_count": int(df[target].nunique()) if target else None,
        "leakage_columns_present": leakage_present,
    }
    return df, summary


def inspect_raw_datasets() -> tuple[dict[str, pd.DataFrame], dict]:
    frames = {}
    summaries = {}
    for filename in RAW_FILES:
        frame, summary = inspect_csv(RAW_DIR / filename)
        frames[filename] = frame
        summaries[filename] = summary

    relationships = []
    for left_name in RAW_FILES:
        for right_name in RAW_FILES:
            if left_name >= right_name:
                continue
            left = frames[left_name]
            right = frames[right_name]
            common_columns = sorted(set(left.columns) & set(right.columns))
            relationships.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "same_columns": set(left.columns) == set(right.columns),
                    "common_column_count": len(common_columns),
                    "common_columns": common_columns,
                    "same_shape": left.shape == right.shape,
                    "duplicates_or_subsets": False,
                    "reason": (
                        "Schemas differ, so these files are not row-level "
                        "duplicates/subsets of one another."
                    )
                    if set(left.columns) != set(right.columns)
                    else "Schemas match; row-level subset check is needed.",
                }
            )

    return frames, {
        "raw_files": summaries,
        "relationships": relationships,
    }


def clean_primary(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing_columns = [
        col for col in [TARGET_COLUMN, *FINAL_FEATURES] if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "primary.csv is missing required project columns: "
            + ", ".join(missing_columns)
        )

    before_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    required_columns = [TARGET_COLUMN, *FINAL_FEATURES]
    missing_required_rows = int(df[required_columns].isna().any(axis=1).sum())
    df = df.dropna(subset=required_columns).copy()

    for feature in NUMERIC_FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
    invalid_numeric_rows = int(df[NUMERIC_FEATURES].isna().any(axis=1).sum())
    df = df.dropna(subset=NUMERIC_FEATURES).copy()

    present_leakage = [col for col in LEAKAGE_COLUMNS if col in df.columns]
    model_df = df[[TARGET_COLUMN, *FINAL_FEATURES]].copy()
    model_df["source_row_id"] = df.index.astype(int)

    summary = {
        "raw_rows": int(before_rows),
        "duplicate_rows_removed": duplicate_rows,
        "rows_removed_for_missing_required_values": missing_required_rows,
        "rows_removed_for_invalid_numeric_values": invalid_numeric_rows,
        "leakage_columns_removed": present_leakage,
        "final_rows": int(len(model_df)),
        "final_feature_count": len(FINAL_FEATURES),
        "target_column": TARGET_COLUMN,
        "crop_class_count": int(model_df[TARGET_COLUMN].nunique()),
    }
    return model_df, summary


def fit_transform_split(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    train_df, test_df = train_test_split(
        model_df,
        test_size=0.2,
        random_state=SEED,
        stratify=model_df[TARGET_COLUMN],
    )
    train_df = train_df.sort_values("source_row_id").reset_index(drop=True)
    test_df = test_df.sort_values("source_row_id").reset_index(drop=True)

    encoders_dir = PROCESSED_DIR / "encoders"
    encoders_dir.mkdir(parents=True, exist_ok=True)

    categorical_encoders = {}
    train_features = train_df[FINAL_FEATURES].copy()
    test_features = test_df[FINAL_FEATURES].copy()

    for feature in CATEGORICAL_FEATURES:
        encoder = LabelEncoder()
        train_values = train_features[feature].astype(str)
        test_values = test_features[feature].astype(str)
        encoder.fit(train_values)
        unseen = sorted(set(test_values) - set(encoder.classes_))
        if unseen:
            raise ValueError(
                f"Test split contains unseen {feature} categories: {unseen}"
            )
        train_features[feature] = encoder.transform(train_values)
        test_features[feature] = encoder.transform(test_values)
        categorical_encoders[feature] = encoder
        joblib.dump(encoder, encoders_dir / f"{feature.lower()}_encoder.pkl")

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df[TARGET_COLUMN].astype(str))
    missing_target_classes = sorted(
        set(test_df[TARGET_COLUMN].astype(str)) - set(label_encoder.classes_)
    )
    if missing_target_classes:
        raise ValueError(
            "Test split contains crop classes absent from training: "
            + ", ".join(missing_target_classes)
        )

    y_train = pd.Series(
        label_encoder.transform(train_df[TARGET_COLUMN].astype(str)),
        name="target",
    )
    y_test = pd.Series(
        label_encoder.transform(test_df[TARGET_COLUMN].astype(str)),
        name="target",
    )
    joblib.dump(label_encoder, PROCESSED_DIR / "label_encoder.pkl")

    scaler = StandardScaler()
    scaler.fit(train_features[FINAL_FEATURES])
    X_train = pd.DataFrame(
        scaler.transform(train_features[FINAL_FEATURES]),
        columns=FINAL_FEATURES,
    )
    X_test = pd.DataFrame(
        scaler.transform(test_features[FINAL_FEATURES]),
        columns=FINAL_FEATURES,
    )
    joblib.dump(scaler, PROCESSED_DIR / "scaler.pkl")

    X_train.insert(0, "source_row_id", train_df["source_row_id"].astype(int))
    X_test.insert(0, "source_row_id", test_df["source_row_id"].astype(int))

    y_train_df = pd.DataFrame(
        {
            "source_row_id": train_df["source_row_id"].astype(int),
            "target": y_train,
            "crop_label": train_df[TARGET_COLUMN].astype(str),
        }
    )
    y_test_df = pd.DataFrame(
        {
            "source_row_id": test_df["source_row_id"].astype(int),
            "target": y_test,
            "crop_label": test_df[TARGET_COLUMN].astype(str),
        }
    )

    artifact_summary = {
        "feature_columns": FINAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "scaler": "StandardScaler",
        "categorical_encoder": "LabelEncoder fitted per categorical feature on training split only",
        "target_encoder": "LabelEncoder fitted on training split only",
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "crop_class_count": int(len(label_encoder.classes_)),
        "crop_classes": list(label_encoder.classes_),
    }

    train_meta = train_df[["source_row_id", "SOIL", TARGET_COLUMN]].rename(
        columns={"SOIL": "SOIL_raw", TARGET_COLUMN: "crop_label"}
    )
    test_meta = test_df[["source_row_id", "SOIL", TARGET_COLUMN]].rename(
        columns={"SOIL": "SOIL_raw", TARGET_COLUMN: "crop_label"}
    )
    metadata = {
        "train_meta": train_meta,
        "test_meta": test_meta,
        "categorical_encoders": categorical_encoders,
        "label_encoder": label_encoder,
        "artifact_summary": artifact_summary,
    }
    return X_train, X_test, y_train_df, y_test_df, metadata


def save_splits(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)
    write_json(PROCESSED_DIR / "feature_columns.json", FINAL_FEATURES)


def _shard_frame(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    train_meta: pd.DataFrame,
) -> pd.DataFrame:
    frame = X_train.merge(y_train, on="source_row_id", validate="one_to_one")
    frame = frame.merge(
        train_meta[["source_row_id", "SOIL_raw"]],
        on="source_row_id",
        validate="one_to_one",
    )
    return frame[["source_row_id", *FINAL_FEATURES, "target", "crop_label", "SOIL_raw"]]


def generate_iid_shards(
    shard_df: pd.DataFrame,
    client_count: int,
) -> list[pd.DataFrame]:
    splitter = StratifiedKFold(
        n_splits=client_count,
        shuffle=True,
        random_state=SEED,
    )
    shards = []
    X_dummy = np.zeros(len(shard_df))
    y = shard_df["target"].to_numpy()
    for _, client_indices in splitter.split(X_dummy, y):
        client = shard_df.iloc[client_indices].sort_values("source_row_id")
        shards.append(client.reset_index(drop=True))
    return shards


def generate_soil_non_iid_shards(
    shard_df: pd.DataFrame,
    client_count: int,
) -> list[pd.DataFrame]:
    soil_counts = shard_df["SOIL_raw"].value_counts()
    if len(soil_counts) < client_count:
        raise ValueError(
            f"Cannot make {client_count} soil-dominant clients from "
            f"{len(soil_counts)} soil categories."
        )

    assignments = [[] for _ in range(client_count)]
    client_sizes = [0 for _ in range(client_count)]

    for soil, count in soil_counts.sort_values(ascending=False).items():
        client_idx = min(range(client_count), key=lambda idx: (client_sizes[idx], idx))
        assignments[client_idx].append(soil)
        client_sizes[client_idx] += int(count)

    shards = []
    for soils in assignments:
        client = shard_df[shard_df["SOIL_raw"].isin(soils)]
        client = client.sort_values(["SOIL_raw", "source_row_id"])
        shards.append(client.reset_index(drop=True))
    return shards


def write_shards(
    shards: list[pd.DataFrame],
    partition: str,
    client_count: int,
) -> list[dict]:
    out_dir = PROCESSED_DIR / "client_shards" / partition / f"{client_count}_clients"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, shard in enumerate(shards, start=1):
        shard_path = out_dir / f"client_{index:02d}.csv"
        shard.to_csv(shard_path, index=False)
        soil_distribution = {
            str(key): int(value)
            for key, value in shard["SOIL_raw"].value_counts().sort_index().items()
        }
        records.append(
            {
                "partition": partition,
                "client_count": client_count,
                "client_id": f"client_{index:02d}",
                "sample_count": int(len(shard)),
                "soil_distribution": json.dumps(soil_distribution, sort_keys=True),
                "crop_class_count": int(shard["target"].nunique()),
                "path": str(shard_path.relative_to(PROJECT_ROOT)),
            }
        )
    return records


def validate_shards(
    partition: str,
    client_count: int,
    shards: list[pd.DataFrame],
    train_ids: set[int],
    test_ids: set[int],
    valid_targets: set[int],
) -> dict:
    expected_columns = ["source_row_id", *FINAL_FEATURES, "target", "crop_label", "SOIL_raw"]
    seen_ids = []
    sample_counts = []

    for index, shard in enumerate(shards, start=1):
        if shard.empty:
            raise ValueError(f"{partition} {client_count} client {index} is empty.")
        if list(shard.columns) != expected_columns:
            raise ValueError(
                f"{partition} {client_count} client {index} has unexpected schema."
            )
        ids = shard["source_row_id"].astype(int).tolist()
        seen_ids.extend(ids)
        sample_counts.append(len(ids))
        invalid_targets = set(shard["target"].astype(int)) - valid_targets
        if invalid_targets:
            raise ValueError(
                f"{partition} {client_count} client {index} has invalid targets."
            )

    seen_counter = Counter(seen_ids)
    duplicate_ids = [key for key, value in seen_counter.items() if value > 1]
    if duplicate_ids:
        raise ValueError(
            f"{partition} {client_count} has duplicate records across clients."
        )
    if set(seen_ids) != train_ids:
        missing = len(train_ids - set(seen_ids))
        extra = len(set(seen_ids) - train_ids)
        raise ValueError(
            f"{partition} {client_count} does not cover training exactly "
            f"(missing={missing}, extra={extra})."
        )
    leaked_test = set(seen_ids) & test_ids
    if leaked_test:
        raise ValueError(
            f"{partition} {client_count} contains {len(leaked_test)} test records."
        )

    return {
        "partition": partition,
        "client_count": client_count,
        "client_files": len(shards),
        "sample_counts": sample_counts,
        "total_samples": int(sum(sample_counts)),
        "no_empty_clients": True,
        "no_duplicate_records_across_clients": True,
        "no_test_records_inside_client_shards": True,
        "every_training_record_assigned_once": True,
        "feature_schema": FINAL_FEATURES,
    }


def generate_client_shards(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    metadata: dict,
) -> tuple[list[dict], list[dict]]:
    shard_df = _shard_frame(X_train, y_train, metadata["train_meta"])
    train_ids = set(X_train["source_row_id"].astype(int))
    test_ids = set(metadata["test_meta"]["source_row_id"].astype(int))
    valid_targets = set(y_train["target"].astype(int))

    distribution_rows = []
    validation_records = []

    for client_count in CLIENT_COUNTS:
        iid_shards = generate_iid_shards(shard_df, client_count)
        distribution_rows.extend(write_shards(iid_shards, "iid", client_count))
        validation_records.append(
            validate_shards(
                "iid", client_count, iid_shards, train_ids, test_ids, valid_targets
            )
        )

        non_iid_shards = generate_soil_non_iid_shards(shard_df, client_count)
        distribution_rows.extend(write_shards(non_iid_shards, "non_iid", client_count))
        validation_records.append(
            validate_shards(
                "non_iid",
                client_count,
                non_iid_shards,
                train_ids,
                test_ids,
                valid_targets,
            )
        )

    pd.DataFrame(distribution_rows).to_csv(
        PROCESSED_DIR / "client_distribution.csv",
        index=False,
    )
    return distribution_rows, validation_records


def run_pipeline() -> dict:
    frames, dataset_summary = inspect_raw_datasets()
    primary_df = frames["primary.csv"]
    model_df, cleaning_summary = clean_primary(primary_df)

    X_train, X_test, y_train, y_test, metadata = fit_transform_split(model_df)
    save_splits(X_train, X_test, y_train, y_test)

    distribution_rows, validation_records = generate_client_shards(
        X_train,
        y_train,
        metadata,
    )

    preprocessing_summary = {
        "source_file": "data/raw/primary.csv",
        "processed_dir": str(PROCESSED_DIR.relative_to(PROJECT_ROOT)),
        "seed": SEED,
        "split": "80:20 stratified by CROPS",
        "cleaning": cleaning_summary,
        "artifacts": metadata["artifact_summary"],
        "leakage_columns_not_used": LEAKAGE_COLUMNS,
        "final_features_used": FINAL_FEATURES,
        "shard_validation": validation_records,
        "non_iid_method": (
            "Training records are grouped by raw SOIL category. Soil groups "
            "are assigned deterministically, largest first, to the currently "
            "smallest client. This keeps each soil category intact, creates "
            "soil-dominant clients, avoids empty clients, and assigns every "
            "training record exactly once. Test records are never used."
        ),
    }
    dataset_summary["selected_roles"] = {
        "primary.csv": "authoritative model training source",
        "analysis.csv": "available for EDA/statistical crop-yield analysis; not used for model training",
        "validation.csv": "held-out external-style validation candidate; not used for FL client training",
    }
    dataset_summary["primary_cleaned"] = cleaning_summary
    dataset_summary["generated_outputs"] = {
        "splits": [
            "data/processed/X_train.csv",
            "data/processed/X_test.csv",
            "data/processed/y_train.csv",
            "data/processed/y_test.csv",
        ],
        "client_distribution_rows": len(distribution_rows),
    }

    write_json(PROCESSED_DIR / "dataset_summary.json", dataset_summary)
    write_json(PROCESSED_DIR / "preprocessing_summary.json", preprocessing_summary)

    return preprocessing_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate AgriFL preprocessing artifacts and client shards."
    )
    parser.parse_args()
    summary = run_pipeline()
    print("AgriFL preprocessing complete.")
    print(f"Processed directory: {summary['processed_dir']}")
    print(f"Train rows: {summary['artifacts']['train_size']}")
    print(f"Test rows: {summary['artifacts']['test_size']}")
    print(f"Crop classes: {summary['artifacts']['crop_class_count']}")
    print("Generated IID and soil-based Non-IID shards for 5, 10, and 20 clients.")


if __name__ == "__main__":
    main()
